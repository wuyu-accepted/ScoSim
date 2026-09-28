"""Bounded research source backed by the existing authorized HTTPS transport."""
from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
import os
import re
from pathlib import Path
import time

from .public_corpus import digest
from .corpus_interaction import atomic_json
from .yulan_scale import TurnEnvelope


class CorpusBudgetExceeded(RuntimeError):
    """Proven pre-dispatch budget refusal, distinct from an unknown outcome."""


class CorpusProvider:
    def __init__(self, plan, directory: Path, *, key_file=None, transport=None, replace_known_once=False, format_repair_limit=0):
        self.plan, self.directory = plan, directory
        self.replace_known_once = replace_known_once
        if format_repair_limit not in (0,1):
            raise ValueError('At most one explicit format repair per slot')
        self.format_repair_limit = format_repair_limit
        directory.mkdir(parents=True, exist_ok=True)
        binding = directory / 'provider-plan.json'
        if binding.exists() and json.loads(binding.read_text(encoding='utf-8')) != plan:
            raise ValueError('Provider plan changed; create a new explicitly budgeted experiment')
        atomic_json(binding, plan)
        self.request_limit = plan['request_limit']
        amendment = directory / 'budget-amendment.json'
        if amendment.exists():
            change = json.loads(amendment.read_text(encoding='utf-8'))
            if (change.get('original_budget_sha256') != digest(plan)
                or change.get('token_limit') != plan['token_limit']
                or type(change.get('request_limit')) is not int
                or change['request_limit'] < plan['request_limit'] or not change.get('reason')):
                raise ValueError('Invalid explicit budget amendment')
            self.request_limit = change['request_limit']
        if plan['request_limit'] < 1 or plan['token_limit'] < 1 or plan['max_output_tokens'] < 1:
            raise ValueError('Invalid Provider limits')
        if transport is None:
            # Imported lazily so data preparation and unit tests need no credentials.
            from backend.campus_pulse.yulan_scale_live_runtime import VectorEngineV7Transport, vectorengine_transport_fingerprint
            key = Path(key_file).read_text(encoding='utf-8-sig').strip()
            # A key file is a credential, not executable configuration.
            if not key.startswith('sk-') or any(c.isspace() for c in key):
                raise ValueError('Credential file must contain exactly one API key')
            env_name = 'CAMPUS_CORPUS_PRIVATE_KEY'
            config = dict(endpoint=plan['endpoint'], model_name=plan['model'], timeout_seconds=45,
                          safe_preconnect_retry_limit=0, local_https_proxy=None)
            old = os.environ.get(env_name)
            try:
                os.environ[env_name] = key
                transport = VectorEngineV7Transport(**config, api_key_env=env_name,
                    expected_fingerprint=vectorengine_transport_fingerprint(**config))
            finally:
                if old is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = old
        self.transport = transport

    def records(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.directory.glob('request-*.json'))]

    def summary(self):
        rows = self.records()
        return {'requests': len(rows), 'tokens': sum(r.get('tokens', 0) for r in rows),
                'reserved_tokens': sum(r.get('reserved', 0) for r in rows if r['status'] in {'in_flight','unknown'}),
                'statuses': {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}},
                'elapsed_seconds': round(sum(r.get('seconds', 0) for r in rows), 3),
                'request_limit': self.request_limit, 'token_limit': self.plan['token_limit']}

    def request_body(self, system, prompt):
        return {'model': self.plan['model'], 'stream': False, 'reasoning_effort': 'none',
                'max_tokens': self.plan['max_output_tokens'], 'response_format': {'type': 'json_object'},
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]}

    def never_dispatched(self, rendered):
        identity = digest({'request': self.request_body(rendered.system_prompt,rendered.agent_prompt),'stage':'resident'})
        rows = self.records()
        return (not any(r['status'] in {'in_flight','unknown'} for r in rows)
                and not any(r['request_sha256'] == identity for r in rows))

    def acquire(self, system, prompt, *, stage, prompt_sha256=None):
        lock = self.directory / 'provider.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError('Provider source active or stale lock requires reconciliation') from exc
        try:
            os.close(fd)
            return self._acquire(system, prompt, stage=stage, prompt_sha256=prompt_sha256)
        finally:
            lock.unlink()

    def _acquire(self, system, prompt, *, stage, prompt_sha256):
        if 'json' not in (system + '\n' + prompt).lower():
            raise ValueError('JSON response mode requires an explicit JSON instruction before dispatch')
        request = self.request_body(system,prompt)
        identity = digest({'request': request, 'stage': stage})
        path = self.directory / f'request-{identity}.json'
        rows = self.records()
        if any(r['status'] in {'in_flight', 'unknown'} for r in rows):
            raise RuntimeError('Unresolved Provider outcome; no retries or new requests allowed')
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['status'] == 'known_failure' and self.replace_known_once:
                # Explicit, appended replacement, never rewriting the failed attempt.
                path = self.directory / f'request-{identity}-replacement-1.json'
                if path.exists():
                    record = json.loads(path.read_text(encoding='utf-8'))
                else:
                    record = None
            if record is None:
                pass
            elif record['status'] != 'received':
                raise RuntimeError('Previously settled failure; replacement exhausted or disabled')
            else:
                saved = record['envelope']
                saved['attempt_records'] = tuple(saved.get('attempt_records', ()))
                return TurnEnvelope(**saved)
        # Conservative UTF-8 byte bound + framing + max output, for byte-tokenizing models.
        reservation = len(json.dumps(request, ensure_ascii=False).encode('utf-8')) + 512 + self.plan['max_output_tokens']
        consumed = sum(r.get('tokens', r.get('reserved', 0)) for r in rows)
        if len(rows) >= self.request_limit or consumed + reservation > self.plan['token_limit']:
            raise CorpusBudgetExceeded('Global request/token budget exhausted before dispatch')
        if reservation > self.plan['slot_reservation']:
            raise RuntimeError('Prompt exceeds frozen per-slot token reservation')
        record = {'status': 'in_flight', 'stage': stage, 'request_sha256': identity, 'reserved': reservation}
        atomic_json(path, record)
        start = time.monotonic()
        try:
            response = self.transport.invoke(request, request_occurrence=1)
        except Exception as exc:
            status = re.search(r'HTTP (\d{3})', str(exc))
            record.update(status='known_failure' if type(exc).__name__ == 'ProviderKnownFailure' else 'unknown',
                          seconds=time.monotonic()-start, diagnostic=type(exc).__name__,
                          http_status=int(status.group(1)) if status else None)
            atomic_json(path, record)
            raise RuntimeError(f"Provider stopped: {record['status']} ({record['diagnostic']})") from None
        record['seconds'] = time.monotonic() - start
        try:
            usage = response['usage']
            total, inp, out = usage['total_tokens'], usage['prompt_tokens'], usage['completion_tokens']
            if any(type(v) is not int or v < 0 for v in (total, inp, out)) or total != inp + out:
                raise ValueError('Invalid token accounting')
            content = response['choices'][0]['message']['content']
            if not isinstance(content, str) or response['model'] != self.plan['model']:
                raise ValueError('Model/content mismatch')
            envelope = TurnEnvelope('resident', prompt_sha256 or identity,
                hashlib.sha256(content.encode()).hexdigest(), 'authorized_live_llm', content,
                response['model'], prompt_tokens=inp, completion_tokens=out, total_tokens=total,
                provider_calls=1, provider_attempts=1, provider_billed_tokens=total)
            record.update(status='received', tokens=total, envelope=asdict(envelope))
        except (KeyError, IndexError, TypeError, ValueError):
            record.update(status='invalid_response', response_sha256=digest(response))
            atomic_json(path, record)
            raise RuntimeError('Invalid Provider response; persisted diagnostic, no retry') from None
        atomic_json(path, record)
        if total > reservation:
            raise RuntimeError('Provider token usage exceeded reservation; stop')
        return envelope

    def __call__(self, rendered, max_tokens_per_slot):
        if max_tokens_per_slot != self.plan['slot_reservation']:
            raise ValueError('Runner/source reservation mismatch')
        return self.acquire(rendered.system_prompt, rendered.agent_prompt, stage='resident',
                            prompt_sha256=rendered.prompt_sha256)

    def repair(self, rendered, original, error):
        if self.format_repair_limit != 1 or original.format_repairs:
            raise RuntimeError('Format repair disabled or already used')
        # Only the known length contract may be repaired by this policy. No new
        # action, stance, target, claim or particle delta can be substituted.
        if not str(error).startswith('resident visible_text exceeds'):
            raise RuntimeError('Failure is outside the approved length-repair policy')
        original_row = json.loads(original.content)
        request = {'decision': original_row, 'error': str(error),
                   'instruction': 'Return the same JSON object. Shorten visible_text only to at most 110 characters, '
                                  'preserving its meaning. Keep every other field exactly unchanged.'}
        repaired = self.acquire('You repair JSON length without changing the decision.',
            json.dumps(request,ensure_ascii=False),stage='length_repair',prompt_sha256=rendered.prompt_sha256)
        replacement_row = json.loads(repaired.content)
        normalization = 'none'
        if (set(replacement_row) == {'decision'} or
                set(replacement_row) == {'decision','error','instruction'}
                and replacement_row['error'] == request['error']
                and replacement_row['instruction'] == request['instruction']):
            # A provider may echo the request wrapper. Unwrap only this exact
            # wrapper; never alter its generated decision or synthesize text.
            replacement_row = replacement_row['decision']
            normalization = 'exact_echo_wrapper_removed'
        if {k:v for k,v in replacement_row.items() if k != 'visible_text'} != {k:v for k,v in original_row.items() if k != 'visible_text'}:
            raise ValueError('Length repair changed decision fields')
        return replace(repaired, content=json.dumps(replacement_row,ensure_ascii=False),
                       prompt_tokens=original.prompt_tokens+repaired.prompt_tokens,
                       completion_tokens=original.completion_tokens+repaired.completion_tokens,
                       total_tokens=original.total_tokens+repaired.total_tokens,
                       provider_calls=2,provider_attempts=2,format_repairs=1,
                       provider_billed_tokens=original.total_tokens+repaired.total_tokens,
                       attempt_records=({'original_response_sha256': original.response_sha256,
                                         'repair_response_sha256': repaired.response_sha256,
                                         'normalization': normalization,
                                         'repair_policy': 'visible_text_length_only'},))
