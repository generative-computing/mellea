"""Comprehensive security tests for mellea thread security features."""

import pytest
from mellea.stdlib.base import CBlock, ModelOutputThunk, ChatContext
from mellea.security import (
    AccessType,
    SecLevel, 
    SecurityMetadata, 
    SecurityError, 
    privileged, 
    sanitize, 
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
        sec_level = SecLevel.none()
        assert sec_level.level_type == "none"
        assert sec_level.is_safe()
        assert not sec_level.is_tainted()
        assert not sec_level.is_classified()
    
    def test_sec_level_tainted_by(self):
        """Test SecLevel.tainted_by() creates tainted level."""
        source = CBlock("source content")
        sec_level = SecLevel.tainted_by(source)
        assert sec_level.level_type == "tainted_by"
        assert not sec_level.is_safe()
        assert sec_level.is_tainted()
        assert not sec_level.is_classified()
        assert sec_level.get_taint_source() is source
    
    def test_sec_level_classified(self):
        """Test SecLevel.classified() creates classified level."""
        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"
        
        access = TestAccess()
        sec_level = SecLevel.classified(access)
        assert sec_level.level_type == "classified"
        assert sec_level.is_safe("admin")
        assert not sec_level.is_safe("user")
        assert not sec_level.is_safe(None)
        assert not sec_level.is_tainted()
        assert sec_level.is_classified()


class TestCBlockSecurity:
    """Test CBlock security functionality."""
    
    def test_cblock_mark_tainted(self):
        """Test marking CBlock as tainted."""
        cblock = CBlock("test content")
        cblock.mark_tainted()
        
        assert "_security" in cblock._meta
        assert isinstance(cblock._meta["_security"], SecurityMetadata)
        assert cblock._meta["_security"].is_tainted()
        assert not cblock.is_safe()
    
    def test_cblock_mark_tainted_by_source(self):
        """Test marking CBlock as tainted by another source."""
        source = CBlock("source content")
        cblock = CBlock("test content")
        cblock.mark_tainted(source)
        
        assert cblock._meta["_security"].is_tainted()
        assert cblock._meta["_security"].get_taint_source() is source
    
    def test_cblock_default_safe(self):
        """Test that CBlock defaults to safe when no security metadata."""
        cblock = CBlock("test content")
        assert cblock.is_safe()
    
    def test_cblock_with_classified_metadata(self):
        """Test CBlock with classified security metadata."""
        class TestAccess(AccessType[str]):
            def has_access(self, entitlement: str | None) -> bool:
                return entitlement == "admin"
        
        access = TestAccess()
        sec_level = SecLevel.classified(access)
        security_meta = SecurityMetadata(sec_level)
        
        cblock = CBlock("classified content", meta={"_security": security_meta})
        
        assert cblock.is_safe("admin")
        assert not cblock.is_safe("user")
        assert not cblock.is_safe(None)


class TestSanitizeDeclassify:
    """Test sanitize and declassify functions."""
    
    def test_sanitize_creates_new_object(self):
        """Test that sanitize creates a new object without mutating original."""
        original = CBlock("test content")
        original.mark_tainted()
        
        sanitized = sanitize(original)
        
        # Objects are different
        assert original is not sanitized
        assert id(original) != id(sanitized)
        
        # Content is preserved
        assert original.value == sanitized.value
        
        # Security levels are different
        assert not original.is_safe()
        assert sanitized.is_safe()
        assert sanitized._meta["_security"].sec_level.level_type == "none"
        
        # Original is unchanged
        assert original._meta["_security"].is_tainted()
    
    def test_declassify_creates_new_object(self):
        """Test that declassify creates a new object without mutating original."""
        original = CBlock("test content")
        original.mark_tainted()
        
        declassified = declassify(original)
        
        # Objects are different
        assert original is not declassified
        assert id(original) != id(declassified)
        
        # Content is preserved
        assert original.value == declassified.value
        
        # Security levels are different
        assert not original.is_safe()
        assert declassified.is_safe()
        assert declassified._meta["_security"].sec_level.level_type == "none"
        
        # Original is unchanged
        assert original._meta["_security"].is_tainted()
    
    def test_sanitize_preserves_other_metadata(self):
        """Test that sanitize preserves other metadata."""
        original = CBlock("test content", meta={"custom": "value", "other": 123})
        original.mark_tainted()
        
        sanitized = sanitize(original)
        
        assert sanitized._meta["custom"] == "value"
        assert sanitized._meta["other"] == 123
        assert sanitized._meta["_security"].sec_level.level_type == "none"


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
    
    def test_privileged_accepts_sanitized_input(self):
        """Test that privileged functions accept sanitized input."""
        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"
        
        tainted_cblock = CBlock("tainted content")
        tainted_cblock.mark_tainted()
        sanitized_cblock = sanitize(tainted_cblock)
        
        result = safe_function(sanitized_cblock)
        assert result == "Processed: tainted content"
    
    def test_privileged_rejects_tainted_input(self):
        """Test that privileged functions reject tainted input."""
        @privileged
        def safe_function(cblock: CBlock) -> str:
            return f"Processed: {cblock.value}"
        
        tainted_cblock = CBlock("tainted content")
        tainted_cblock.mark_tainted()
        
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
        
        tainted_cblock = CBlock("tainted content")
        tainted_cblock.mark_tainted()
        
        with pytest.raises(SecurityError, match="argument 'data'"):
            safe_function(data=tainted_cblock)


class TestTaintSources:
    """Test taint source computation."""
    
    def test_taint_sources_from_tainted_action(self):
        """Test taint sources from tainted action."""
        action = CBlock("tainted action")
        action.mark_tainted()
        
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
        tainted_cblock = CBlock("tainted context")
        tainted_cblock.mark_tainted()
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
        taint_source = CBlock("taint source")
        taint_source.mark_tainted()
        
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=[taint_source],
            meta={"custom": "value"}
        )
        
        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert "_security" in mot._meta
        assert mot._meta["_security"].is_tainted()
        assert not mot.is_safe()
        assert mot._meta["_security"].get_taint_source() is taint_source
    
    def test_from_generation_without_taint_sources(self):
        """Test ModelOutputThunk.from_generation without taint sources."""
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=None,
            meta={"custom": "value"}
        )
        
        assert mot.value == "generated content"
        assert mot._meta["custom"] == "value"
        assert "_security" in mot._meta
        assert mot._meta["_security"].sec_level.level_type == "none"
        assert mot.is_safe()
    
    def test_from_generation_empty_taint_sources(self):
        """Test ModelOutputThunk.from_generation with empty taint sources."""
        mot = ModelOutputThunk.from_generation(
            value="generated content",
            taint_sources=[],
            meta={"custom": "value"}
        )
        
        assert mot._meta["_security"].sec_level.level_type == "none"
        assert mot.is_safe()


class TestSecurityIntegration:
    """Test integration between security components."""
    
    def test_security_flow_through_generation(self):
        """Test security metadata flows through generation pipeline."""
        # Create tainted input
        tainted_input = CBlock("user input")
        tainted_input.mark_tainted()
        
        # Simulate generation with taint sources
        sources = taint_sources(tainted_input, None)
        mot = ModelOutputThunk.from_generation(
            value="model response",
            taint_sources=sources
        )
        
        # Verify output is tainted
        assert not mot.is_safe()
        assert mot._meta["_security"].is_tainted()
        
        # Sanitize the output
        safe_mot = sanitize(mot)
        assert safe_mot.is_safe()
        assert safe_mot._meta["_security"].sec_level.level_type == "none"
        
        # Verify original is unchanged
        assert not mot.is_safe()
    
    def test_privileged_function_with_generated_content(self):
        """Test privileged function with generated content."""
        @privileged
        def process_response(mot: ModelOutputThunk) -> str:
            return f"Processed: {mot.value}"
        
        # Generate tainted content
        taint_source = CBlock("taint source")
        taint_source.mark_tainted()
        
        mot = ModelOutputThunk.from_generation(
            value="tainted response",
            taint_sources=[taint_source]
        )
        
        # Privileged function should reject tainted content
        with pytest.raises(SecurityError):
            process_response(mot)
        
        # Sanitize and try again
        safe_mot = sanitize(mot)
        result = process_response(safe_mot)
        assert result == "Processed: tainted response"
