import pytest
from tinygrad import Tensor
from src.algs import *

class MockModel:
  def __init__(self, predicted_actions):
    self.predicted_actions = predicted_actions

  def __call__(self, h, c, obs, tl, bl, prev_actions):
    return self.predicted_actions, None, (None, None)

def test_bc_accuracy():
    B, T, D = 3, 4, 5  
    predicted_actions = Tensor.randint(B, T, D, low=0, high=22)  
    action_targets = Tensor.randint(B, T, low=0, high=22)

    model = MockModel(predicted_actions)

    accuracy = bc_accuracy(model, h=None, c=None, obs=None, tl=None, bl=None, action_targets=action_targets, prev_actions=None)

    correct_predictions = (predicted_actions == action_targets).float()
    expected_accuracy = correct_predictions.mean().item() * 100

    assert accuracy == pytest.approx(expected_accuracy), f"Expected {expected_accuracy}, but got {accuracy}"


