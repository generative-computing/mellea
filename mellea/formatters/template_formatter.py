"""Template Formatter."""

import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any

import jinja2

from ..backends import ModelIdentifier, SimpleLRUCache
from ..core import CBlock, Component, FancyLogger, TemplateRepresentation
from .chat_formatter import ChatFormatter


class TemplateFormatter(ChatFormatter):
    """Formatter that uses jinja2 templates."""

    def __init__(
        self,
        model_id: str | ModelIdentifier,
        *,
        template_path: str = "",
        use_template_cache: bool = True,
    ):
        """A TemplateFormatter use jinja2 templates.

        Args:
            model_id: Describes the model for which templates will be looked up. Should match the template dir structure.
            template_path: Specify an alternate location where templates can be found. Will be preferred over all other template dirs even if a less exact match is found.
            use_template_cache: Cache the location of the most recent templates so that future lookups don't need to be performed. Set to false if you plan on changing the model_id or template_path after the TemplateFormatter has been created.
        """
        self.model_id = model_id
        self._template_path: str = template_path

        self._use_template_cache: bool = use_template_cache

        # Key: obj.__class__.__name___ -> Value: jinja2.Template
        self._template_cache = SimpleLRUCache(10) if self._use_template_cache else None

    def _stringify(
        self,
        c: (
            str
            | Component
            | CBlock
            | Iterable
            | Mapping
            | TemplateRepresentation
            | None
        ),
    ) -> Any:
        """A recursive function that ensures an object is stringified.

        For strings and CBlocks, this is just getting their value.
        For Components, this means traversing the fields of their template args dictionary and stringifying each part.

        Iterables and Mappings should only be encountered as parts of a component's template args. We process each item in them while maintaining the structure.
        """
        if c is None:
            return ""

        match c:
            case str():
                return c

            case CBlock():
                assert c.value is not None
                return c.value

            case Component():
                representation = c.format_for_llm()
                if type(representation) is str:
                    return representation
                else:
                    assert isinstance(representation, TemplateRepresentation)
                    stringified_template_args = {}
                    for key, val in representation.args.items():
                        stringified_template_args[key] = self._stringify(val)

                    if representation.obj is None:
                        FancyLogger.get_logger().warning(
                            f"template formatter encountered a TemplateRepresentation with no obj when stringifying {c.__class__}; setting obj to {c}"
                        )
                        representation.obj = c
                    return self._load_template(representation).render(
                        stringified_template_args
                    )

            case c if isinstance(c, Mapping):
                stringified_template_args = {}
                for key, val in c.items():
                    stringified_template_args[key] = self._stringify(val)

                return stringified_template_args

            case c if isinstance(c, Iterable) and not isinstance(c, Mapping):
                stringified_list = []
                for val in c:
                    stringified_list.append(self._stringify(val))

                return stringified_list

            case _:
                FancyLogger.get_logger().warning(
                    f"formatter encountered an unexpected type in _stringify; using str() on {c.__class__}"
                )
                return str(c)

    def print(self, c: Component | CBlock) -> str:
        """Uses a jinja2 template to pretty-print components."""
        return self._stringify(c)

    def _load_template(self, repr: TemplateRepresentation) -> jinja2.Template:
        """This method makes an attempt at auto-loading a Template for the Component.

        Iterates over template order in the template representation. If it finds
        a '*' it will check for a template with the name of the object in 'obj'.
        Once it finds a template, it stops searching the list.

        To find a template, it looks for it in the following places in order:
        1. the user provided template location
        2. the component's package or the mellea package if none is found

        Raises:
            Exception: If there's an unexpected jinja error or the template cannot be found.
        """
        if repr.template:
            return jinja2.Environment().from_string(repr.template)  # type: ignore

        if repr.template_order is None:
            FancyLogger.get_logger().warning(
                f"using template formatter for {repr.obj.__class__.__name__} but no template order was provided, defaulting to class name"
            )
            repr.template_order = ["*"]

        # If using a cache, check the cache for the loaded template first.
        if self._use_template_cache and self._template_cache:
            res = self._template_cache.get(repr.obj.__class__.__name__)
            if res:
                return res

        qualified_tmpl_name = (
            ""  # Includes some basic path info: `prompts/default/Instruction.jinja2`.
        )
        loader: jinja2.PackageLoader | jinja2.FileSystemLoader | None = None

        # Iterate over all possible template names to find a match.
        for t_name in repr.template_order:
            if t_name == "*":
                tmpl_name = repr.obj.__class__.__name__ + ".jinja2"
            else:
                tmpl_name = t_name + ".jinja2"

            if self._template_path != "":
                # Look for the template at the user specified path if it exists.
                loader = jinja2.FileSystemLoader(self._template_path)
                qualified_tmpl_name = self._get_template(self._template_path, tmpl_name)

            if qualified_tmpl_name == "":
                # Try to get the package name to load the templates from it.
                # Default to Mellea if one can't be found.
                package = _get_package_name(repr.obj.__class__.__module__)
                try:
                    loader = jinja2.PackageLoader(package)
                except Exception as e:
                    if package != "mellea":
                        # Mellea should always be available.
                        loader = jinja2.PackageLoader("mellea")
                    else:
                        raise ValueError(
                            f"could not find package for obj: {repr.obj}, exception: {e}"
                        )

                qualified_tmpl_name = self._get_template(
                    loader._template_root, tmpl_name
                )

            # If the qualitified template name isn't an empty string,
            # we've found a template and its loader.
            if qualified_tmpl_name != "":
                break

        if qualified_tmpl_name == "":
            raise ValueError(
                f"could not find template candidate for {repr.obj}, searched: {repr.template_order}"
            )

        try:
            # If we get here, we know the loader is populated and we have found a file that
            # contains the template for the object.
            env = jinja2.Environment(
                loader=loader, autoescape=jinja2.select_autoescape()
            )
            tmpl = env.get_template(qualified_tmpl_name)

        except (jinja2.TemplateNotFound, ValueError) as tnf:
            # This shouldn't happen since we specifically walk the directory tree to get the best template, but
            # there's a chance some file system issue happens.
            raise Exception(
                f"Could not find a template for {repr.obj} using template {qualified_tmpl_name}: {tnf}"
            )

        except Exception as e:
            raise Exception(f"Unexpected template error for component {repr.obj}: {e}")

        if self._use_template_cache and self._template_cache:
            self._template_cache.put(repr.obj.__class__.__name__, tmpl)
        return tmpl

    def _get_template(self, root_path: str, template_name: str) -> str:
        """Attempts to walk the provided directory structure to find the best matching template.

        Prefers the most exact match (meaning deepest directory tree) by:
        1. Looking for directory names matching the model name
        2. Looking in the `prompts/default/` directory for a template

        Assumes that only one directory at each level matches.
        """
        # Simplify the provided model to make matching easier.
        model_id = self._get_model_id()
        simplified_model_id = _simplify_model_string(model_id)

        # Traverse the directory structure to find the most specific template.
        path_offset = len(root_path)  # Used to get the template name used by Jinja2.
        candidate_template = ""
        for root, dirs, files in os.walk(top=root_path, topdown=True):
            # Only look at the default templates if a candidate hasn't yet been chosen.
            # Otherwise, we already have a more specific template.
            is_default = root.rsplit("/")[-1].lower() == "default"
            candidate_found = candidate_template != ""
            if not (is_default and candidate_found):
                for file in files:
                    if file.lower() == template_name.lower():
                        template_path = root[path_offset:].lstrip(os.path.sep)
                        candidate_template = os.path.join(template_path, file).replace(
                            os.path.sep, "/"
                        )

            # Only traverse file paths that are in the model id or are prompts/default.
            for dir in dirs:
                if dir.lower() == "prompts" or dir.lower() == "default":
                    continue

                # Simplify the directory as well for matching names.
                if _simplify_model_string(dir.lower()) not in simplified_model_id:
                    dirs.remove(dir)

        return candidate_template

    def _get_model_id(self) -> str:
        """Gets a string representation of the formatter's model id."""
        model_id = self.model_id
        if type(model_id) is str:
            return model_id

        assert isinstance(model_id, ModelIdentifier), (
            "model_id was neither a `str` nor `ModelIdentifier`"
        )

        # Go through the ModelIdentifier's fields, find one that can be matched against.
        for field in fields(model_id):
            val = getattr(model_id, field.name)
            if val is not None and val != "":
                return val

        return ""  # Cannot match against any model identifiers. Will ultimately use default.


def _simplify_model_string(input: str) -> str:
    """Removes special chars from the given string and lower cases it to simplify matching."""
    remove_chars = r"[-\.\/:,]"
    return re.sub(remove_chars, "", input.lower())


def _get_package_name(module: str) -> str:
    """Given a module, attempts to get the package and verifies it exists."""
    package = module.split(".")[0]

    if sys.modules.get(package, None) is None:
        package = ""

    return package
