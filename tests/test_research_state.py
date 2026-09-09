import json
from copy import deepcopy

import pytest

from wmloop.control.mechanism_hypothesis import build_mechanism_hypothesis, MechanismHypothesisError
from wmloop.control.research_state import (
    ResearchJournal, ResearchStateError, build_research_state,
    transition_research_state, validate_research_state,
)


def initial():
    return build_research_state(goal='Improve held-out rollout consistency', metrics=['error'], budget_gpu_hours=2)


def test_transitions_preserve_goal_and_reject_authority_changes():
    state = initial()
    next_state = transition_research_state(state, phase='hypothesize', reason='Probe observations available')
    selected = transition_research_state(next_state, phase='select', reason='Compare feasible candidates')
    assert selected['goal'] == state['goal']
    assert selected['revision'] == 2
    assert selected['last_transition']['parent_state_id'] == next_state['state_id']
    assert state['phase'] == 'observe'
    for updates, error in [({'budget': {}}, 'REQUIRES_LEDGER'), ({'goal': {}}, 'IMMUTABLE'), ({'representation_version': 'new'}, 'IMMUTABLE')]:
        with pytest.raises(ResearchStateError, match=error):
            transition_research_state(state, phase='hypothesize', reason='Attempt mutation', updates=updates)
    with pytest.raises(ResearchStateError, match='TRANSITION_INVALID'):
        transition_research_state(state, phase='verify', reason='Skip research stages')
    state['goal']['description'] = 'A different goal'
    with pytest.raises(ResearchStateError, match='DIGEST_MISMATCH'):
        validate_research_state(state)


def test_journal_resume_retains_history_and_rejects_different_inputs(tmp_path):
    state = initial()
    journal = ResearchJournal(tmp_path, input_hash='a'*64, initial=state)
    next_state = journal.advance('hypothesize', 'Observations received')
    resumed = ResearchJournal.open_or_create(tmp_path, input_hash='a'*64, initial=state)
    assert resumed.read() == next_state
    assert json.loads((tmp_path/'research-history'/f"{state['state_id']}.json").read_text()) == state
    with pytest.raises(ResearchStateError, match='INPUT_MISMATCH'):
        ResearchJournal(tmp_path, input_hash='b'*64, initial=state)
    assert resumed.read() == next_state


def test_journal_rejects_symlink_binding_and_missing_binding(tmp_path):
    external = tmp_path/'external.json'
    external.write_text('{}')
    directory = tmp_path/'journal'
    directory.mkdir()
    (directory/'research-binding.json').symlink_to(external)
    with pytest.raises(ResearchStateError, match='PATH_INVALID'):
        ResearchJournal(directory, input_hash='a'*64, initial=initial())
    (directory/'research-binding.json').unlink()
    (directory/'research-state.json').write_text(json.dumps(initial()))
    with pytest.raises(ResearchStateError, match='BINDING_MISSING'):
        ResearchJournal(directory, input_hash='a'*64, initial=initial())
    assert external.read_text() == '{}'


def test_hypotheses_must_have_valid_mechanism_contracts():
    hypothesis = build_mechanism_hypothesis(
        target_failure='Rollout error grows over time', causal_assumption='Context eviction removes stable scene information',
        target_touchpoint=['history selector'], predicted_observation='Retaining anchors reduces late rollout error',
        falsification_test='No gain against paired baseline with anchors removed',
    )
    state = transition_research_state(initial(), phase='hypothesize', reason='Propose competing explanations', updates={'hypotheses': [hypothesis]})
    validate_research_state(state)
    bad = deepcopy(hypothesis)
    bad['predicted_observation'] = 'A different predicted observation'
    with pytest.raises(MechanismHypothesisError, match='DIGEST_MISMATCH'):
        transition_research_state(state, phase='select', reason='Choose explanation', updates={'hypotheses':[bad]})
    with pytest.raises(MechanismHypothesisError):
        transition_research_state(state, phase='select', reason='Choose explanation', updates={'hypotheses':[{'claim':'untyped guess'}]})


def test_journal_rejects_replaced_snapshot_from_an_unrelated_history(tmp_path):
    state = initial()
    journal = ResearchJournal(tmp_path, input_hash='a'*64, initial=state)
    journal.advance('hypothesize', 'Record actual observations')
    other = transition_research_state(state, phase='hypothesize', reason='Different observations')
    graft = transition_research_state(other, phase='select', reason='Select from unrelated observations')
    journal.path.write_text(json.dumps(graft))
    with pytest.raises(ResearchStateError, match='HISTORY_MISSING'):
        ResearchJournal(tmp_path, input_hash='a'*64, initial=state)


def test_journal_does_not_silently_reset_missing_current_snapshot(tmp_path):
    state = initial()
    journal = ResearchJournal(tmp_path, input_hash='a'*64, initial=state)
    journal.advance('hypothesize', 'Record actual observations')
    journal.path.unlink()
    with pytest.raises(ResearchStateError, match='SNAPSHOT_MISSING'):
        ResearchJournal(tmp_path, input_hash='a'*64, initial=state)


def test_stop_conditions_cannot_be_weakened_by_transition():
    with pytest.raises(ResearchStateError, match='IMMUTABLE_BINDING'):
        transition_research_state(initial(), phase='hypothesize', reason='Weaken campaign constraints', updates={'stop_conditions':[]})
