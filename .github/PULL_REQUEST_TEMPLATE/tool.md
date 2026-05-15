# Tool PR Checklist

Use this checklist when adding or modifying tools in `mellea/stdlib/tools/`.

### Protocol Compliance
- [ ] Ensure compatibility with existing backends and providers
    - For most tools being added as functions, this means that calling `convert_function_to_tool` works

### Integration
- [ ] Tool exported in `mellea/stdlib/tools/__init__.py` or, if you are adding a library of tools, from your sub-module
