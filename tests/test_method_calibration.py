import json
from pathlib import Path
import pytest
from test_mechanism_hypothesis import compile_proposal, proposal, explanation
from wmloop.control.method_calibration import run_method_calibration, MethodCalibrationError

ROOT = Path(__file__).resolve().parents[1]

def test_calibration_executes_declared_checks_and_records_failure(tmp_path):
    compilation = tmp_path / 'compilation'
    compile_proposal(proposal(explanation()), compilation)
    result = run_method_calibration(compilation_root=compilation, output_root=tmp_path/'calibration')
    assert result['state'] == 'passed'
    assert result['checks'][0]['state'] == 'passed'
    assert (tmp_path/'calibration'/'calibration.json').is_file()

def test_calibration_does_not_treat_failed_check_as_success(tmp_path):
    value = proposal(explanation())
    value['tests'][0]['command'] = ['python', '-c', 'raise SystemExit(3)']
    compilation = tmp_path / 'compilation'
    compile_proposal(value, compilation)
    result = run_method_calibration(compilation_root=compilation, output_root=tmp_path/'calibration')
    assert result['state'] == 'failed'
    assert result['checks'][0]['returncode'] == 3

def test_calibration_requires_fresh_ready_compilation(tmp_path):
    with pytest.raises(MethodCalibrationError, match='INPUT_INVALID'):
        run_method_calibration(compilation_root=tmp_path/'missing', output_root=tmp_path/'calibration')
