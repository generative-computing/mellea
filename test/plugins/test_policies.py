"""Tests for hook payload policies."""

from mellea.plugins.policies import MELLEA_HOOK_PAYLOAD_POLICIES


class TestPolicies:
    def test_policies_loaded(self):
        """Policy table should be populated when contextforge is available."""
        assert len(MELLEA_HOOK_PAYLOAD_POLICIES) > 0

    def test_session_pre_init_policy(self):
        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("session_pre_init")
        assert policy is not None
        assert "backend_name" in policy.writable_fields
        assert "model_id" in policy.writable_fields
        assert "model_options" in policy.writable_fields
        assert "backend_kwargs" in policy.writable_fields

    def test_component_pre_create_policy(self):
        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("component_pre_create")
        assert policy is not None
        assert "description" in policy.writable_fields
        assert "requirements" in policy.writable_fields
        assert "images" in policy.writable_fields

    def test_generation_pre_call_policy(self):
        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("generation_pre_call")
        assert policy is not None
        assert "model_options" in policy.writable_fields
        assert "format" in policy.writable_fields

    def test_observe_only_hooks_absent(self):
        """Observe-only hooks should not have entries in the policy table."""
        assert "session_post_init" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "session_reset" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "session_cleanup" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "component_post_error" not in MELLEA_HOOK_PAYLOAD_POLICIES
        assert "sampling_iteration" not in MELLEA_HOOK_PAYLOAD_POLICIES

    def test_validation_policies(self):
        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("validation_pre_check")
        assert policy is not None
        assert "requirements" in policy.writable_fields

        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("validation_post_check")
        assert policy is not None
        assert "results" in policy.writable_fields
        assert "all_passed" in policy.writable_fields

    def test_sampling_policies(self):
        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("sampling_loop_start")
        assert policy is not None
        assert "loop_budget" in policy.writable_fields

        policy = MELLEA_HOOK_PAYLOAD_POLICIES.get("sampling_repair")
        assert policy is not None
        assert "repair_action" in policy.writable_fields
        assert "repair_context" in policy.writable_fields
