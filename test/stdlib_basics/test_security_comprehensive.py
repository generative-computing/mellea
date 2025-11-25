"""Comprehensive security tests for mellea thread security features."""

import pytest
from mellea.stdlib.base import CBlock, ModelOutputThunk, ChatContext
from mellea.security import (
    AccessType,
    SecLevel, 
    SecurityMetadata, 
    SecurityError, 
    privileged, 
    declassify,
    taint_sources
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
        assert sec_level.get_taint_source() is source
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
        assert isinstance(cblock.sec_level, SecurityMetadata)
        assert cblock.sec_level.is_tainted()
        assert not cblock.sec_level.is_classified()
        assert cblock.sec_level.get_access_type() is None
    
    def test_cblock_mark_tainted_by_source(self):
        """Test marking CBlock as tainted by another source."""
        source = CBlock("source content")
        cblock = CBlock("test content", sec_level=SecLevel.tainted_by(source))
        
        assert cblock.sec_level.is_tainted()
        assert cblock.sec_level.get_taint_source() is source
    
    def test_cblock_default_safe(self):
        """Test that CBlock defaults to safe when no security metadata."""
        cblock = CBlock("test content")
        assert cblock.sec_level is None or (not cblock.sec_level.is_tainted() and not cblock.sec_level.is_classified())
    
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
        assert declassified.sec_level.sec_level.level_type == SecLevelType.NONE
        
        # Original is unchanged
        assert original.sec_level.is_tainted()
    
    def test_declassify_preserves_other_metadata(self):
        """Test that declassify preserves other metadata."""
        from mellea.security.core import SecLevelType
        original = CBlock("test content", meta={"custom": "value", "other": 123}, sec_level=SecLevel.tainted_by(None))
        
        declassified = declassify(original)
        
        assert declassified._meta["custom"] == "value"
        assert declassified._meta["other"] == 123
        assert declassified.sec_level.sec_level.level_type == SecLevelType.NONE


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
        security_meta = SecurityMetadata(sec_level)
        
        classified_cblock = CBlock("classified content", meta={"_security": security_meta})
        
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


class TestModelOutputThunkSecurity:
    """Test ModelOutputThunk security functionality."""
    
    def test_from_generation_with_taint_sources(self):
        """Test ModelOutputThunk.from_generation with taint sources."""
        from mellea.security.core import SecLevelType
        taint_source = CBlock("taint source", sec_level=SecLevel.tainted_by(None))
        
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=[taint_source],
            meta={"custom": "value"}
        )
        
        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert mot.sec_level is not None
        assert mot.sec_level.is_tainted()
        assert not mot.sec_level.is_classified()
        assert mot.sec_level.get_taint_source() is taint_source
    
    def test_from_generation_without_taint_sources(self):
        """Test ModelOutputThunk.from_generation without taint sources."""
        from mellea.security.core import SecLevelType
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=None,
            meta={"custom": "value"}
        )
        
        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert mot.sec_level is not None
        assert mot.sec_level.sec_level.level_type == SecLevelType.NONE
        assert not mot.sec_level.is_tainted()
        assert not mot.sec_level.is_classified()
    
    def test_from_generation_empty_taint_sources(self):
        """Test ModelOutputThunk.from_generation with empty taint sources."""
        from mellea.security.core import SecLevelType
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=[],
            meta={"custom": "value"}
        )
        
        assert mot.sec_level.sec_level.level_type == SecLevelType.NONE
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
        mot = ModelOutputThunk.from_generation(
            value="model response",
            taint_sources=sources
        )
        
        # Verify output is tainted
        assert mot.sec_level.is_tainted()
        
        # Declassify the output
        safe_mot = declassify(mot)
        assert not safe_mot.sec_level.is_tainted()
        assert not safe_mot.sec_level.is_classified()
        assert safe_mot.sec_level.sec_level.level_type == SecLevelType.NONE
        
        # Verify original is unchanged
        assert mot.sec_level.is_tainted()
    
    def test_privileged_function_with_generated_content(self):
        """Test privileged function with generated content."""
        @privileged
        def process_response(mot: ModelOutputThunk) -> str:
            return f"Processed: {mot.value}"
        
        # Generate tainted content
        taint_source = CBlock("taint source", sec_level=SecLevel.tainted_by(None))
        
        mot = ModelOutputThunk.from_generation(
            value="tainted response",
            taint_sources=[taint_source]
        )
        
        # Privileged function should reject tainted content
        with pytest.raises(SecurityError):
            process_response(mot)
        
        # Declassify and try again
        safe_mot = declassify(mot)
        result = process_response(safe_mot)
        assert result == "Processed: tainted response"
