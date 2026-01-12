"""Comprehensive security tests for mellea thread security features."""

import pytest
from mellea.stdlib.base import CBlock, ModelOutputThunk, ChatContext, SimpleComponent
from mellea.stdlib.instruction import Instruction
from mellea.security import (
    AccessType,
    SecLevel,
    SecurityError,
    privileged,
    declassify,
    taint_sources,
)


class TestAccessType:
    """Test AccessType functionality."""

    def test_access_type_interface(self):
        """Test that AccessType is an abstract base class."""
        with pytest.raises(TypeError):
            AccessType()  # Should not be instantiable directly

    def test_access_type_implementation(self):
        """Test implementing AccessType."""

        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"

        access = TestAccess()
        assert access.has_access("admin")
        assert not access.has_access("user")
        assert not access.has_access(None)


class TestSecLevel:
    """Test SecLevel functionality."""

    def test_sec_level_none(self):
        """Test SecLevel.none() creates safe level."""
        from mellea.security.core import SecLevelType

        sec_level = SecLevel.none()
        assert sec_level.level_type == SecLevelType.NONE
        assert not sec_level.is_tainted()
        assert not sec_level.is_classified()
        assert sec_level.get_access_type() is None

    def test_sec_level_tainted_by(self):
        """Test SecLevel.tainted_by() creates tainted level."""
        from mellea.security.core import SecLevelType

        source = CBlock("source content")
        sec_level = SecLevel.tainted_by(source)
        assert sec_level.level_type == SecLevelType.TAINTED_BY
        assert sec_level.is_tainted()
        assert not sec_level.is_classified()
        assert sec_level.get_taint_sources() == [source]
        assert sec_level.get_access_type() is None

    def test_sec_level_classified(self):
        """Test SecLevel.classified() creates classified level."""
        from mellea.security.core import SecLevelType

        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"

        access = TestAccess()
        sec_level = SecLevel.classified(access)
        assert sec_level.level_type == SecLevelType.CLASSIFIED
        assert not sec_level.is_tainted()
        assert sec_level.is_classified()
        assert sec_level.get_access_type() is access
        assert sec_level.get_access_type().has_access("admin")
        assert not sec_level.get_access_type().has_access("user")
        assert not sec_level.get_access_type().has_access(None)


class TestCBlockSecurity:
    """Test CBlock security functionality."""

    def test_cblock_mark_tainted(self):
        """Test marking CBlock as tainted."""
        cblock = CBlock("test content", sec_level=SecLevel.tainted_by(None))

        assert cblock.sec_level is not None
        assert cblock.sec_level.is_tainted()
        assert not cblock.sec_level.is_classified()
        assert cblock.sec_level.get_access_type() is None

    def test_cblock_mark_tainted_by_source(self):
        """Test marking CBlock as tainted by another source."""
        source = CBlock("source content")
        cblock = CBlock("test content", sec_level=SecLevel.tainted_by(source))

        assert cblock.sec_level.is_tainted()
        assert cblock.sec_level.get_taint_sources() == [source]

    def test_cblock_default_safe(self):
        """Test that CBlock defaults to safe when no security metadata."""
        cblock = CBlock("test content")
        assert cblock.sec_level is None or (
            not cblock.sec_level.is_tainted() and not cblock.sec_level.is_classified()
        )

    def test_cblock_with_classified_metadata(self):
        """Test CBlock with classified security metadata."""

        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"

        access = TestAccess()
        sec_level = SecLevel.classified(access)

        cblock = CBlock("classified content", sec_level=sec_level)

        assert cblock.sec_level.is_classified()
        access_type = cblock.sec_level.get_access_type()
        assert access_type is not None
        assert access_type.has_access("admin")
        assert not access_type.has_access("user")
        assert not access_type.has_access(None)


class TestDeclassify:
    """Test declassify function."""

    def test_declassify_creates_new_object(self):
        """Test that declassify creates a new object without mutating original."""
        from mellea.security.core import SecLevelType

        original = CBlock("test content", sec_level=SecLevel.tainted_by(None))

        declassified = declassify(original)

        # Objects are different
        assert original is not declassified
        assert id(original) != id(declassified)

        # Content is preserved
        assert original.value == declassified.value

        # Security levels are different
        assert original.sec_level.is_tainted()
        assert not declassified.sec_level.is_tainted()
        assert not declassified.sec_level.is_classified()
        assert declassified.sec_level.level_type == SecLevelType.NONE

        # Original is unchanged
        assert original.sec_level.is_tainted()

    def test_declassify_preserves_other_metadata(self):
        """Test that declassify preserves other metadata."""
        from mellea.security.core import SecLevelType

        original = CBlock(
            "test content",
            meta={"custom": "value", "other": 123},
            sec_level=SecLevel.tainted_by(None),
        )

        declassified = declassify(original)

        assert declassified._meta["custom"] == "value"
        assert declassified._meta["other"] == 123
        assert declassified.sec_level.level_type == SecLevelType.NONE


class TestPrivilegedDecorator:
    """Test @privileged decorator functionality."""

    def test_privileged_accepts_safe_input(self):
        """Test that privileged functions accept safe input."""

        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"

        # CBlock with no security metadata defaults to safe
        safe_cblock = CBlock("safe content")

        result = safe_function(safe_cblock)
        assert result == "Processed: safe content"

    def test_privileged_accepts_declassified_input(self):
        """Test that privileged functions accept declassified input."""

        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"

        tainted_cblock = CBlock("tainted content", sec_level=SecLevel.tainted_by(None))
        declassified_cblock = declassify(tainted_cblock)

        result = safe_function(declassified_cblock)
        assert result == "Processed: tainted content"

    def test_privileged_rejects_tainted_input(self):
        """Test that privileged functions reject tainted input."""

        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"

        tainted_cblock = CBlock("tainted content", sec_level=SecLevel.tainted_by(None))

        with pytest.raises(SecurityError, match="requires safe input"):
            safe_function(tainted_cblock)

    def test_privileged_rejects_classified_input(self):
        """Test that privileged functions reject classified input without proper entitlement."""

        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"

        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"

        access = TestAccess()
        sec_level = SecLevel.classified(access)

        classified_cblock = CBlock("classified content", sec_level=sec_level)

        with pytest.raises(SecurityError, match="requires safe input"):
            safe_function(classified_cblock)

    def test_privileged_accepts_no_security_metadata(self):
        """Test that privileged functions accept input with no security metadata."""

        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"

        # CBlock with no security metadata defaults to safe
        cblock = CBlock("content")

        result = safe_function(cblock)
        assert result == "Processed: content"

    def test_privileged_with_kwargs(self):
        """Test privileged function with keyword arguments."""

        @privileged
        def safe_function(data: CBlock, prefix: str = "Processed: ") -> str:
            return f"{prefix}{data.value}"

        tainted_cblock = CBlock("tainted content", sec_level=SecLevel.tainted_by(None))

        with pytest.raises(SecurityError, match="argument 'data'"):
            safe_function(data=tainted_cblock)


class TestTaintSources:
    """Test taint source computation."""

    def test_taint_sources_from_tainted_action(self):
        """Test taint sources from tainted action."""
        action = CBlock("tainted action", sec_level=SecLevel.tainted_by(None))

        sources = taint_sources(action, None)
        assert len(sources) == 1
        assert sources[0] is action

    def test_taint_sources_from_safe_action(self):
        """Test taint sources from safe action."""
        action = CBlock("safe action")
        # No security metadata - defaults to safe

        sources = taint_sources(action, None)
        assert len(sources) == 0

    def test_taint_sources_from_context(self):
        """Test taint sources from context."""
        action = CBlock("safe action")

        # Create context with tainted content
        ctx = ChatContext()
        tainted_cblock = CBlock("tainted context", sec_level=SecLevel.tainted_by(None))
        ctx = ctx.add(tainted_cblock)

        sources = taint_sources(action, ctx)
        assert len(sources) == 1
        assert sources[0] is tainted_cblock

    def test_taint_sources_empty(self):
        """Test taint sources with no tainted content."""
        action = CBlock("safe action")
        ctx = ChatContext()
        safe_cblock = CBlock("safe context")
        # No security metadata - defaults to safe
        ctx = ctx.add(safe_cblock)

        sources = taint_sources(action, ctx)
        assert len(sources) == 0

    def test_taint_sources_from_component_parts(self):
        """Test taint sources from Component parts."""
        # Create Instruction with tainted description
        tainted_desc = CBlock(
            "tainted description", sec_level=SecLevel.tainted_by(None)
        )
        instruction = Instruction(description=tainted_desc)

        sources = taint_sources(instruction, None)
        assert len(sources) == 1
        assert sources[0] is tainted_desc

    def test_taint_sources_from_nested_component_with_tainted_cblocks(self):
        """Test taint sources from nested Components containing tainted CBlocks."""
        # Create tainted CBlocks
        tainted_data = CBlock(
            "sensitive user data", sec_level=SecLevel.tainted_by(None)
        )
        tainted_config = CBlock("secret config", sec_level=SecLevel.tainted_by(None))
        safe_info = CBlock("public info")  # Safe CBlock

        # Create a SimpleComponent with mixed tainted and safe CBlocks
        nested_component = SimpleComponent(
            data=tainted_data, config=tainted_config, info=safe_info
        )

        # Create an Instruction with the nested Component in grounding_context
        instruction = Instruction(
            description="Process the data",
            grounding_context={"context": nested_component},
        )

        # taint_sources should find both tainted CBlocks through the nested Component
        sources = taint_sources(instruction, None)

        # Should find both tainted CBlocks
        assert len(sources) == 2
        assert tainted_data in sources
        assert tainted_config in sources
        assert safe_info not in sources  # Safe CBlock should not be included

    def test_taint_sources_shallow_search_limit(self):
        """Test that shallow search only checks last 5 components."""
        action = CBlock("safe action")

        # Create context with 7 items: tainted at positions 0 and 5
        ctx = ChatContext()
        tainted_early = CBlock("tainted early", sec_level=SecLevel.tainted_by(None))
        ctx = ctx.add(tainted_early)  # Position 0 - outside last 5

        # Add 4 safe items
        for i in range(4):
            ctx = ctx.add(CBlock(f"safe {i}"))

        tainted_late = CBlock("tainted late", sec_level=SecLevel.tainted_by(None))
        ctx = ctx.add(tainted_late)  # Position 5 - within last 5

        # Add one more safe item
        ctx = ctx.add(CBlock("safe final"))  # Position 6

        sources = taint_sources(action, ctx)
        # Should only find tainted_late (position 5), not tainted_early (position 0)
        assert len(sources) == 1
        assert sources[0] is tainted_late


class TestModelOutputThunkSecurity:
    """Test ModelOutputThunk security functionality."""

    def test_from_generation_with_taint_sources(self):
        """Test ModelOutputThunk creation with taint sources."""
        taint_source = CBlock("taint source", sec_level=SecLevel.tainted_by(None))

        sec_level = SecLevel.tainted_by([taint_source])
        mot = ModelOutputThunk(
            value="generated content", sec_level=sec_level, meta={"custom": "value"}
        )

        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert mot.sec_level is not None
        assert mot.sec_level.is_tainted()
        assert not mot.sec_level.is_classified()
        assert mot.sec_level.get_taint_sources() == [taint_source]

    def test_from_generation_without_taint_sources(self):
        """Test ModelOutputThunk creation without taint sources."""
        from mellea.security.core import SecLevelType

        mot = ModelOutputThunk(
            value="generated content",
            sec_level=SecLevel.none(),
            meta={"custom": "value"},
        )

        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert mot.sec_level is not None
        assert mot.sec_level.level_type == SecLevelType.NONE
        assert not mot.sec_level.is_tainted()
        assert not mot.sec_level.is_classified()

    def test_from_generation_empty_taint_sources(self):
        """Test ModelOutputThunk creation with empty taint sources."""
        from mellea.security.core import SecLevelType

        mot = ModelOutputThunk(
            value="generated content",
            sec_level=SecLevel.none(),
            meta={"custom": "value"},
        )

        assert mot.sec_level.level_type == SecLevelType.NONE
        assert not mot.sec_level.is_tainted()
        assert not mot.sec_level.is_classified()


class TestSecurityIntegration:
    """Test integration between security components."""

    def test_security_flow_through_generation(self):
        """Test security metadata flows through generation pipeline."""
        from mellea.security.core import SecLevelType

        # Create tainted input
        tainted_input = CBlock("user input", sec_level=SecLevel.tainted_by(None))

        # Simulate generation with taint sources
        sources = taint_sources(tainted_input, None)
        sec_level = SecLevel.tainted_by(sources) if sources else SecLevel.none()
        mot = ModelOutputThunk(value="model response", sec_level=sec_level)

        # Verify output is tainted
        assert mot.sec_level.is_tainted()

        # Declassify the output
        safe_mot = declassify(mot)
        assert not safe_mot.sec_level.is_tainted()
        assert not safe_mot.sec_level.is_classified()
        assert safe_mot.sec_level.level_type == SecLevelType.NONE

        # Verify original is unchanged
        assert mot.sec_level.is_tainted()

    def test_privileged_function_with_generated_content(self):
        """Test privileged function with generated content."""

        @privileged
        def process_response(mot: ModelOutputThunk) -> str:
            return f"Processed: {mot.value}"

        # Generate tainted content
        taint_source = CBlock("taint source", sec_level=SecLevel.tainted_by(None))

        sec_level = SecLevel.tainted_by([taint_source])
        mot = ModelOutputThunk(value="tainted response", sec_level=sec_level)

        # Privileged function should reject tainted content
        with pytest.raises(SecurityError):
            process_response(mot)

        # Declassify and try again
        safe_mot = declassify(mot)
        result = process_response(safe_mot)
        assert result == "Processed: tainted response"
