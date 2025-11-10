#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# String Processing Nodes for ComfyUI MaraScott Custom Nodes
# Provides text manipulation capabilities for STRING and COMBO inputs
#

from typing import Any, Tuple

from ...utils.constants import get_category
from ...utils.log import log


class StringOperations:
    """Centralized string operation methods for consistency across nodes."""

    @staticmethod
    def apply_operation(text: str, operation: str) -> str:
        """
        Apply a string operation to the given text.

        Args:
            text: The input string to transform
            operation: The operation to apply

        Returns:
            The transformed string
        """
        operations = {
            "none": lambda x: x,
            "uppercase": lambda x: x.upper(),
            "lowercase": lambda x: x.lower(),
            "capitalize": lambda x: x.capitalize(),
            "title": lambda x: x.title(),
            "invert_case": lambda x: x.swapcase(),
            "reverse": lambda x: x[::-1],
            "trim": lambda x: x.strip(),
            "remove_spaces": lambda x: x.replace(" ", ""),
            "remove_extra_spaces": lambda x: " ".join(x.split()),
            "snake_case": lambda x: x.replace(" ", "_").lower(),
            "kebab_case": lambda x: x.replace(" ", "-").lower(),
        }

        if operation in operations:
            return operations[operation](text)
        else:
            log(f"Unknown string operation: {operation}", None, None, "warning")
            return text

    @staticmethod
    def get_operations_list() -> list:
        """Return list of available string operations."""
        return [
            "none",
            "uppercase",
            "lowercase",
            "capitalize",
            "title",
            "invert_case",
            "reverse",
            "trim",
            "remove_spaces",
            "remove_extra_spaces",
            "snake_case",
            "kebab_case"
        ]


class StringProcessor_v1:
    """
    String Processor Node - Process text with optional COMBO input override.

    Features:
    - Direct text input processing
    - Optional COMBO input that can override text input
    - Multiple string operations available
    - Detailed logging and error handling
    """

    NAME = "String Processor"
    SHORTCUT = "sp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Input text to process"
                }),
                "operation": (StringOperations.get_operations_list(), {
                    "default": "none",
                    "tooltip": "String operation to apply to the input text"
                }),
            },
            "optional": {
                "combo_input": ("COMBO", {
                    "tooltip": "Optional COMBO input that will override text input if provided"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "source_text")
    FUNCTION = "process_string"
    CATEGORY = get_category("Utils")
    OUTPUT_NODE = False

    def process_string(self, text: str = "", operation: str = "none", combo_input: Any = None) -> Tuple[str, str]:
        """
        Process input text or COMBO value with the specified string operation.

        Args:
            text: Direct text input
            operation: The string operation to apply
            combo_input: Optional COMBO input that overrides text input

        Returns:
            - processed_text: The text after applying the operation
            - source_text: The source text that was processed
        """
        try:
            # Determine source text - prioritize combo_input if provided
            if combo_input is not None:
                source_text = str(combo_input)
                log(f"Using COMBO input: '{source_text}' (type: {type(combo_input).__name__})", None, None, "debug")
            else:
                source_text = text
                log(f"Using text input: '{source_text}'", None, None, "debug")

            # Apply the string operation
            processed_text = StringOperations.apply_operation(source_text, operation)

            log(f"Applied '{operation}' operation: '{source_text}' -> '{processed_text}'", None, None, "info")

            return (processed_text, source_text)

        except Exception as e:
            error_msg = f"Error processing string with operation '{operation}': {str(e)}"
            log(error_msg, None, None, "error")
            # Return source text as fallback
            fallback_text = str(combo_input) if combo_input is not None else text
            return (fallback_text, fallback_text)


class ComboProcessor_v1:
    """
    COMBO Processor Node - Dedicated COMBO input processing with label override.

    Features:
    - Accepts any COMBO type input
    - Optional string label that overrides COMBO value for processing
    - Multiple string operations available
    - Returns processed text, original COMBO text, and passthrough COMBO
    - Comprehensive logging and error handling
    """

    NAME = "COMBO Processor"
    SHORTCUT = "cp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combo_input": ("COMBO", {
                    "tooltip": "COMBO input that will be converted to string and processed"
                }),
                "operation": (StringOperations.get_operations_list(), {
                    "default": "none",
                    "tooltip": "String operation to apply to the COMBO value or label"
                }),
            },
            "optional": {
                "combo_label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional string label that overrides the COMBO value for processing"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "COMBO")
    RETURN_NAMES = ("processed_text", "combo_text", "passthrough_combo")
    FUNCTION = "process_combo"
    CATEGORY = get_category("Utils")
    OUTPUT_NODE = False

    def process_combo(self, combo_input: Any, operation: str = "none", combo_label: str = "") -> Tuple[str, str, Any]:
        """
        Process COMBO input with optional label override and string operation.

        Args:
            combo_input: COMBO input (any type that can be converted to string)
            operation: The string operation to apply
            combo_label: Optional string label that overrides COMBO value for processing

        Returns:
            - processed_text: Result of applying operation to combo_label or combo_input
            - combo_text: Original COMBO value as string (always from combo_input)
            - passthrough_combo: Original COMBO value unchanged for chaining
        """
        try:
            # Convert COMBO to string for combo_text output
            combo_text = str(combo_input)

            # Determine what text to process - prioritize combo_label if provided
            if combo_label and combo_label.strip():
                text_to_process = combo_label.strip()
                log(f"Processing with combo_label: '{text_to_process}' (original combo: '{combo_text}')", None, None, "info")
            else:
                text_to_process = combo_text
                log(f"Processing COMBO input: '{combo_text}' (type: {type(combo_input).__name__})", None, None, "debug")

            # Apply the string operation
            processed_text = StringOperations.apply_operation(text_to_process, operation)

            log(f"Applied '{operation}' operation: '{text_to_process}' -> '{processed_text}'", None, None, "info")

            return (processed_text, combo_text, combo_input)

        except Exception as e:
            error_msg = f"Error processing COMBO with operation '{operation}': {str(e)}"
            log(error_msg, None, None, "error")
            # Fallback to original COMBO value
            fallback_text = str(combo_input) if combo_input is not None else ""
            return (fallback_text, fallback_text, combo_input)


class TextJoiner_v1:
    """
    Text Joiner Node - Join multiple text inputs with configurable separator.

    Features:
    - Join 2-5 text inputs
    - Configurable separator
    - Skip empty inputs option
    - Trim inputs option
    """

    NAME = "Text Joiner"
    SHORTCUT = "tj"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "First text input"
                }),
                "separator": ("STRING", {
                    "default": " ",
                    "multiline": False,
                    "tooltip": "Separator to use between text inputs"
                }),
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip empty text inputs when joining"
                }),
                "trim_inputs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trim whitespace from inputs before joining"
                }),
            },
            "optional": {
                "text2": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Second text input"
                }),
                "text3": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Third text input"
                }),
                "text4": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Fourth text input"
                }),
                "text5": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Fifth text input"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("joined_text", "parts_count")
    FUNCTION = "join_texts"
    CATEGORY = get_category("Utils")
    OUTPUT_NODE = False

    def join_texts(self, text1: str, separator: str = " ", skip_empty: bool = True,
                   trim_inputs: bool = True, text2: str = "", text3: str = "",
                   text4: str = "", text5: str = "") -> Tuple[str, int]:
        """
        Join multiple text inputs with specified separator.

        Args:
            text1: First text input (required)
            separator: Separator to use between texts
            skip_empty: Whether to skip empty inputs
            trim_inputs: Whether to trim inputs before processing
            text2-5: Additional optional text inputs

        Returns:
            - joined_text: The joined text string
            - parts_count: Number of parts that were joined
        """
        try:
            # Collect all text inputs
            texts = [text1, text2, text3, text4, text5]

            # Process texts based on options
            processed_texts = []
            for text in texts:
                if trim_inputs:
                    text = text.strip()

                if skip_empty and not text:
                    continue

                processed_texts.append(text)

            # Join the texts
            joined_text = separator.join(processed_texts)
            parts_count = len(processed_texts)

            log(f"Joined {parts_count} text parts with separator '{separator}': '{joined_text}'", None, None, "info")

            return (joined_text, parts_count)

        except Exception as e:
            error_msg = f"Error joining texts: {str(e)}"
            log(error_msg, None, None, "error")
            return (text1, 1)
