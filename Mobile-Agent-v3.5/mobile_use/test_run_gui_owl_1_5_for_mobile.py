"""Tests for bounded model-action parsing and retry (issue #268)."""

import json
import unittest
from unittest import mock

from run_gui_owl_1_5_for_mobile import parse_action, predict_action_with_retry


def tool_call(action='click'):
    return '<tool_call>\n' + json.dumps({
        'name': 'mobile_use', 'arguments': {'action': action}
    }) + '\n</tool_call>'


class ToolCallRetryTest(unittest.TestCase):

    def test_parses_valid_tool_call_without_required_newline(self):
        action = parse_action('Action: click\n<tool_call>' + json.dumps({
            'name': 'mobile_use', 'arguments': {'action': 'click'}
        }) + '</tool_call>')
        self.assertEqual(action['arguments']['action'], 'click')

    def test_missing_or_invalid_action_is_rejected(self):
        for output in ('', None, 'No tool call', '<tool_call>{bad}</tool_call>',
                       '<tool_call>[]</tool_call>', '<tool_call>{}</tool_call>',
                       '<tool_call>{"arguments":{}}</tool_call>'):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    parse_action(output)

    def test_retry_once_then_accept_valid_action(self):
        model = mock.Mock()
        model.predict_mm.side_effect = [
            ('', None, None), (tool_call(), None, None)
        ]
        output, action = predict_action_with_retry(model, [{'role': 'user'}])
        self.assertEqual(output, tool_call())
        self.assertEqual(action['arguments']['action'], 'click')
        self.assertEqual(model.predict_mm.call_count, 2)

    def test_two_invalid_responses_never_return_an_action(self):
        model = mock.Mock()
        model.predict_mm.return_value = ('bad response', None, None)
        output, action = predict_action_with_retry(model, [], max_attempts=2)
        self.assertIsNone(output)
        self.assertIsNone(action)
        self.assertEqual(model.predict_mm.call_count, 2)

    def test_valid_response_is_not_retried(self):
        model = mock.Mock()
        model.predict_mm.return_value = (tool_call('wait'), None, None)
        _, action = predict_action_with_retry(model, [])
        self.assertEqual(action['arguments']['action'], 'wait')
        model.predict_mm.assert_called_once()

    def test_request_error_is_not_silently_retried(self):
        model = mock.Mock()
        model.predict_mm.side_effect = RuntimeError('API unavailable')
        with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
            predict_action_with_retry(model, [])
        model.predict_mm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
