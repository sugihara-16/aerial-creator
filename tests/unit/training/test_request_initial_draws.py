import json

import pytest

from amsrr.training.request_initial_draws import draw_initial_case
from amsrr.utils.hashing import hash_file


def _job(tmp_path):
    record = tmp_path / 'record.json'
    record.write_text(json.dumps({'dataset_split': 'validation'}))
    return dict(source_hashes={}, configuration_hashes={}, checkpoints={},
        checkpoint_identities={}, require_training_split=True,
        case=dict(record=str(record), record_sha256=hash_file(record), split='train'))


def test_parallel_draw_rejects_validation_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match='training split'):
        draw_initial_case(_job(tmp_path))


def test_parallel_draw_checks_current_input_bytes(tmp_path):
    job = _job(tmp_path)
    job['case']['record_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='input changed'):
        draw_initial_case(job)
