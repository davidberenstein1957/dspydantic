"""Human-In-The-Loop (HITL) evaluation UI components.

This module provides GUI components for human-in-the-loop evaluation,
including loading windows and popup dialogs for reviewing and editing
extracted outputs.
"""

import json
import re
from collections.abc import Callable
from io import BytesIO
from typing import Any

import dspy
from pydantic import BaseModel

from dspydantic.extractor import apply_optimized_descriptions
from dspydantic.types import Example
from dspydantic.utils import convert_images_to_dspy_images


class HitlManager:
    """Manager for Human-In-The-Loop evaluation UI components."""

    def __init__(self, optimizer: Any) -> None:
        """Initialize HITL manager with reference to optimizer.

        Args:
            optimizer: The PydanticOptimizer instance.
        """
        self.optimizer = optimizer
        self._hitl_window: Any = None
        self._hitl_window_centered: bool = False
        self._loading_window: Any = None
        self._hitl_photo_refs: list[Any] = []

    def show_loading_window(
        self, evaluation_num: int | None = None, total_evaluations: int | None = None
    ) -> None:
        """Show a loading window while processing the next evaluation.

        Args:
            evaluation_num: Current evaluation number (1-based).
            total_evaluations: Total number of evaluations.
        """
        import tkinter as tk
        from tkinter import ttk

        loading_root = tk.Tk()
        loading_root.title("Processing...")

        # Set progress message
        if evaluation_num is not None and total_evaluations is not None:
            if evaluation_num < total_evaluations:
                message = (
                    f"Processing evaluation {evaluation_num + 1} of {total_evaluations}...\n"
                    "Please wait while the next evaluation is prepared."
                )
            else:
                message = "Processing final results...\nPlease wait."
        elif evaluation_num is not None:
            message = (
                f"Processing evaluation {evaluation_num + 1}...\n"
                "Please wait while the next evaluation is prepared."
            )
        else:
            message = "Processing next evaluation...\nPlease wait."

        main_frame = ttk.Frame(loading_root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        label = ttk.Label(
            main_frame,
            text=message,
            font=("Arial", 11),
            justify=tk.CENTER,
        )
        label.pack(pady=10)

        # Add a simple progress indicator (animated dots)
        progress_label = ttk.Label(
            main_frame,
            text="⏳",
            font=("Arial", 20),
        )
        progress_label.pack(pady=5)

        loading_root.geometry("400x150")
        loading_root.resizable(False, False)

        # Center window
        loading_root.update_idletasks()
        width = loading_root.winfo_width()
        height = loading_root.winfo_height()
        x = (loading_root.winfo_screenwidth() // 2) - (width // 2)
        y = (loading_root.winfo_screenheight() // 2) - (height // 2)
        loading_root.geometry(f"{width}x{height}+{x}+{y}")

        # Make it appear on top
        loading_root.lift()
        loading_root.attributes("-topmost", True)
        loading_root.focus_set()

        # Update to show the window and process events
        loading_root.update_idletasks()
        loading_root.update()

        # Store reference so it can be closed later
        self._loading_window = loading_root

        # Process events periodically to keep window responsive
        # This will be interrupted when the window is destroyed
        def process_events():
            if hasattr(self, "_loading_window") and self._loading_window:
                try:
                    self._loading_window.update()
                    self._loading_window.after(100, process_events)
                except Exception:
                    pass

        loading_root.after(100, process_events)

    def close_loading_window(self) -> None:
        """Close the loading window if it exists."""
        if hasattr(self, "_loading_window") and self._loading_window:
            try:
                self._loading_window.destroy()
            except Exception:
                pass
            self._loading_window = None

    def _convert_to_pil_images(
        self, images: list[str] | list[Any] | None
    ) -> list[Any] | None:
        """Convert base64 strings or other formats to PIL Image objects.

        Args:
            images: List of base64 strings, PIL Image objects, bytes, or dspy.Image objects.

        Returns:
            List of PIL Image objects, or None if no images or PIL not available.
        """
        if not images:
            return None

        try:
            from PIL import Image as PILImage  # noqa: N806
        except ImportError:
            return None

        pil_images = []
        for img_input in images:
            if img_input is None:
                continue

            # Already a PIL Image - use directly
            if isinstance(img_input, PILImage.Image):
                pil_images.append(img_input)
                continue

            # Try to convert to PIL Image
            try:
                # Handle bytes - try to open directly as image
                if isinstance(img_input, bytes):
                    try:
                        img = PILImage.open(BytesIO(img_input))
                        pil_images.append(img)
                        continue
                    except Exception:
                        # Not raw image bytes, try as base64 string
                        try:
                            img_input = img_input.decode("utf-8")
                        except UnicodeDecodeError:
                            continue

                # Handle dspy.Image objects - extract base64 from data URL
                if hasattr(img_input, "url"):
                    url = img_input.url
                    if isinstance(url, str) and url.startswith("data:"):
                        comma_idx = url.find(",")
                        if comma_idx != -1:
                            img_input = url[comma_idx + 1 :]
                        else:
                            continue
                    elif hasattr(img_input, "base64"):
                        img_input = img_input.base64
                    else:
                        continue

                # Ensure we have a string
                if not isinstance(img_input, str):
                    img_input = str(img_input)

                # Strip whitespace
                img_input = img_input.strip()
                if not img_input:
                    continue

                # Remove data URL prefix if present
                if img_input.startswith("data:"):
                    comma_idx = img_input.find(",")
                    if comma_idx != -1:
                        img_input = img_input[comma_idx + 1 :].strip()

                # Decode base64 and open with PIL
                import base64

                try:
                    img_data = base64.b64decode(img_input, validate=True)
                except (TypeError, ValueError):
                    try:
                        img_data = base64.b64decode(img_input)
                    except (TypeError, ValueError):
                        continue

                if len(img_data) == 0:
                    continue

                # Open image with PIL
                img = PILImage.open(BytesIO(img_data))
                pil_images.append(img)

            except Exception:
                # Skip images that can't be converted
                continue

        return pil_images if pil_images else None

    def show_hitl_popup(
        self,
        input_text: str | None,
        images: list[str] | list[Any] | None,
        proposed_output: dict[str, Any],
        evaluation_num: int | None = None,
        total_evaluations: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Show a GUI popup for human-in-the-loop evaluation.

        Reuses the same window across evaluations, updating content in place.

        Args:
            input_text: Input text to display.
            images: List of base64-encoded images, PIL Image objects, or other formats to display.
                   Will be converted to PIL Image objects internally.
            proposed_output: Proposed output JSON to display and allow editing.
            evaluation_num: Current evaluation number (1-based).
            total_evaluations: Total number of evaluations.

        Returns:
            Tuple of (edited_output, was_edited) where edited_output is the final JSON
            and was_edited indicates if the user made changes.
        """
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        try:
            from PIL import Image as PILImage  # noqa: N806
            from PIL import ImageTk as PILImageTk  # noqa: N806
        except ImportError:
            PILImage = None  # type: ignore[assignment]  # noqa: N806
            PILImageTk = None  # type: ignore[assignment]  # noqa: N806

        # Convert all images to PIL Image objects
        pil_images = self._convert_to_pil_images(images) if images else None

        # Check if we have an existing window to reuse
        if self._hitl_window:
            try:
                if self._hitl_window.winfo_exists():
                    root = self._hitl_window
                    # Clear existing content (but keep old photo references until new ones are created)
                    for widget in root.winfo_children():
                        widget.destroy()
                    # Update window title
                    if evaluation_num is not None and total_evaluations is not None:
                        root.title(
                            f"HITL Evaluation - Review and Edit Output "
                            f"({evaluation_num}/{total_evaluations})"
                        )
                    elif evaluation_num is not None:
                        root.title(
                            f"HITL Evaluation - Review and Edit Output "
                            f"(Evaluation {evaluation_num})"
                        )
                    else:
                        root.title("HITL Evaluation - Review and Edit Output")
                else:
                    # Window was destroyed, create new one
                    root = None
            except Exception:
                # Window is invalid, create new one
                root = None
        else:
            root = None

        if root is None:
            # Create main window (not modal) - only on first evaluation
            root = tk.Tk()
            self._hitl_window = root

            # Set progress title
            if evaluation_num is not None and total_evaluations is not None:
                title = (
                    f"HITL Evaluation - Review and Edit Output "
                    f"({evaluation_num}/{total_evaluations})"
                )
            elif evaluation_num is not None:
                title = f"HITL Evaluation - Review and Edit Output (Evaluation {evaluation_num})"
            else:
                title = "HITL Evaluation - Review and Edit Output"

            root.title(title)
            root.geometry("1000x800")

            # Make it the main window (not transient/modal)
            root.lift()
            root.attributes("-topmost", True)
            root.after_idle(root.attributes, "-topmost", False)
        elif evaluation_num is not None:
            title = f"HITL Evaluation - Review and Edit Output (Evaluation {evaluation_num})"
        else:
            title = "HITL Evaluation - Review and Edit Output"

        root.title(title)
        root.geometry("1000x800")

        # Make it the main window (not transient/modal)
        root.lift()
        root.attributes("-topmost", True)
        root.after_idle(root.attributes, "-topmost", False)

        was_edited = False
        edited_output: dict[str, Any] = proposed_output.copy()

        # Create a variable to track when Continue is clicked
        continue_clicked = tk.BooleanVar(value=False)

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Progress notification with progress bar
        if evaluation_num is not None:
            progress_frame = ttk.Frame(main_frame)
            progress_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            progress_frame.columnconfigure(1, weight=1)

            if total_evaluations is not None:
                progress_text = f"Evaluation {evaluation_num} of {total_evaluations}"
                # Calculate progress percentage
                progress_value = (evaluation_num / total_evaluations) * 100
            else:
                progress_text = f"Evaluation {evaluation_num}"
                progress_value = 0  # Unknown total, show indeterminate

            # Progress text label
            progress_label = tk.Label(
                progress_frame,
                text=progress_text,
                font=("Arial", 10, "bold"),
                fg="blue",
            )
            progress_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

            # Create style for green progress bar
            style = ttk.Style()
            style.theme_use("default")
            style.configure(
                "Green.Horizontal.TProgressbar",
                background="green",
                troughcolor="lightgray",
                borderwidth=0,
                lightcolor="green",
                darkcolor="green",
            )

            # Progress bar with green style
            if total_evaluations is not None:
                progress_bar = ttk.Progressbar(
                    progress_frame,
                    style="Green.Horizontal.TProgressbar",
                    mode="determinate",
                    maximum=100,
                    value=progress_value,
                    length=300,
                )
            else:
                progress_bar = ttk.Progressbar(
                    progress_frame,
                    style="Green.Horizontal.TProgressbar",
                    mode="indeterminate",
                    length=300,
                )
                progress_bar.start()

            progress_bar.grid(row=0, column=1, sticky="ew", padx=(0, 10))

            # Percentage label (if we have total)
            if total_evaluations is not None:
                percentage_label = tk.Label(
                    progress_frame,
                    text=f"{int(progress_value)}%",
                    font=("Arial", 9),
                    fg="gray",
                )
                percentage_label.grid(row=0, column=2, sticky=tk.E)

        # Input section
        row_offset = 1 if evaluation_num is not None else 0
        input_label = ttk.Label(main_frame, text="Input:", font=("Arial", 12, "bold"))
        input_label.grid(row=row_offset, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        # Create notebook for input (text/images)
        input_notebook = ttk.Notebook(main_frame)
        input_notebook.grid(row=row_offset + 1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # Track which tab to select by default (images take priority)
        tab_to_select = None

        # Images tab (add first so it's selected by default)
        if pil_images and len(pil_images) > 0:
            images_frame = ttk.Frame(input_notebook, padding="5")
            input_notebook.add(images_frame, text=f"Images ({len(pil_images)})")

            # Create a scrollable container using Frame + Scrollbar
            scroll_container = ttk.Frame(images_frame)
            scroll_container.pack(fill=tk.BOTH, expand=True)

            # Create scrollbar
            images_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical")
            images_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Create canvas for scrolling
            images_canvas = tk.Canvas(
                scroll_container,
                yscrollcommand=images_scrollbar.set,
                bg="white",
            )
            images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            images_scrollbar.config(command=images_canvas.yview)

            # Create frame inside canvas for images
            images_scrollable_frame = ttk.Frame(images_canvas)
            canvas_window = images_canvas.create_window(
                (0, 0), window=images_scrollable_frame, anchor="nw"
            )

            def on_frame_configure(event):
                """Update scroll region when frame size changes."""
                images_canvas.configure(scrollregion=images_canvas.bbox("all"))

            def on_canvas_configure(event):
                """Update canvas window width when canvas size changes."""
                canvas_width = event.width
                images_canvas.itemconfig(canvas_window, width=canvas_width)

            images_scrollable_frame.bind("<Configure>", on_frame_configure)
            images_canvas.bind("<Configure>", on_canvas_configure)

            if PILImage and PILImageTk:
                # Store photo references to prevent garbage collection
                # This is critical - Tkinter will garbage collect images without references
                photo_refs = []
                # pil_images are already PIL Image objects, use them directly
                for i, img in enumerate(pil_images):
                    try:
                        # img is already a PIL Image object from _convert_to_pil_images
                        if not isinstance(img, PILImage.Image):
                            raise ValueError(f"Image {i + 1}: Expected PIL Image object")

                        # Convert to RGB if necessary (for formats like RGBA)
                        if img.mode in ("RGBA", "LA", "P"):
                            # Create a white background
                            rgb_img = PILImage.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "P":
                                img = img.convert("RGBA")
                            mask = img.split()[-1] if img.mode in ("RGBA", "LA") else None
                            rgb_img.paste(img, mask=mask)
                            img = rgb_img
                        elif img.mode != "RGB":
                            img = img.convert("RGB")

                        # Resize if too large
                        max_size = 600
                        if img.width > max_size or img.height > max_size:
                            img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)

                        # Create PhotoImage - MUST keep reference to prevent garbage collection
                        # See: https://stackoverflow.com/questions/74937810/image-not-being-displayed-in-tkinter-python
                        photo = PILImageTk.PhotoImage(img)

                        # SOLUTION 1: Store reference in list (persists beyond function scope)
                        # This list will be stored on the root window to prevent garbage collection
                        photo_refs.append(photo)

                        # SOLUTION 2: Store reference on label itself (internal image variable)
                        # Create label with image
                        img_label = ttk.Label(images_scrollable_frame, image=photo)
                        # Set the internal image variable - this is critical for Tkinter
                        # The label widget maintains a reference to the image through this attribute
                        img_label.image = photo  # type: ignore[attr-defined]

                        # Also store reference on the scrollable frame as backup
                        # This ensures the reference persists even if label is destroyed
                        if not hasattr(images_scrollable_frame, "_image_refs"):
                            images_scrollable_frame._image_refs = []  # type: ignore[attr-defined]
                        images_scrollable_frame._image_refs.append(photo)  # type: ignore

                        img_label.grid(row=i, column=0, padx=5, pady=5)
                    except Exception as e:
                        error_msg = f"Image {i + 1}: Could not display\n{str(e)}"
                        error_label = tk.Label(
                            images_scrollable_frame,
                            text=error_msg,
                            fg="red",
                            wraplength=400,
                            justify=tk.LEFT,
                        )
                        error_label.grid(row=i, column=0, padx=5, pady=5)

                # SOLUTION 1: Store photo references on root window (instance variable)
                # This ensures references persist beyond function scope and prevent
                # garbage collection. Delete old photo references AFTER creating new ones.
                if hasattr(root, "_photo_refs"):
                    delattr(root, "_photo_refs")
                # Store as instance variable on root window - persists for lifetime
                root._photo_refs = photo_refs  # type: ignore[attr-defined]

                # Also store on the manager instance as a backup reference
                # This ensures references persist even if the window is recreated
                self._hitl_photo_refs.extend(photo_refs)
            else:
                no_pil_label = ttk.Label(
                    images_scrollable_frame,
                    text="PIL/Pillow not installed. Install with: uv pip install pillow",
                    foreground="orange",
                )
                no_pil_label.grid(row=0, column=0, padx=5, pady=5)

            # Update canvas scroll region after adding images
            root.update_idletasks()
            images_scrollable_frame.update_idletasks()
            images_canvas.configure(scrollregion=images_canvas.bbox("all"))

            # Images tab is at index 0 if added first
            tab_to_select = 0
        else:
            # No images - clear old photo references if they exist
            if hasattr(root, "_photo_refs"):
                delattr(root, "_photo_refs")

        # Text input tab
        if input_text:
            text_frame = ttk.Frame(input_notebook, padding="5")
            input_notebook.add(text_frame, text="Text")
            text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10, width=80)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert("1.0", input_text)
            text_widget.config(state=tk.DISABLED)
            # Text tab is at index 0 if no images, otherwise index 1
            if tab_to_select is None:
                tab_to_select = 0

        # Select the appropriate tab by default
        if tab_to_select is not None:
            input_notebook.select(tab_to_select)

        # Output section
        output_label = ttk.Label(
            main_frame,
            text="Proposed Output (editable):",
            font=("Arial", 12, "bold"),
        )
        output_label.grid(row=row_offset + 2, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        output_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15, width=80)
        output_text.grid(row=row_offset + 3, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        output_text.insert("1.0", json.dumps(proposed_output, indent=2))

        original_output_text = json.dumps(proposed_output, indent=2)

        def on_output_change(event: Any = None) -> None:
            """Track if output was edited."""
            nonlocal was_edited
            current_text = output_text.get("1.0", tk.END).strip()
            if current_text != original_output_text:
                was_edited = True

        output_text.bind("<KeyRelease>", on_output_change)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row_offset + 4, column=0, columnspan=2, pady=10)

        def continue_eval() -> None:
            """Continue evaluation with edited output."""
            nonlocal edited_output
            try:
                edited_text = output_text.get("1.0", tk.END).strip()
                edited_output = json.loads(edited_text)
                # Signal that Continue was clicked
                continue_clicked.set(True)
            except json.JSONDecodeError as e:
                error_window = tk.Toplevel(root)
                error_window.title("JSON Error")
                error_label = ttk.Label(
                    error_window,
                    text=f"Invalid JSON: {str(e)}\n\nPlease fix the JSON before continuing.",
                    foreground="red",
                    padding="10",
                )
                error_label.pack()
                ok_button = ttk.Button(error_window, text="OK", command=error_window.destroy)
                ok_button.pack(pady=5)

        continue_button = ttk.Button(button_frame, text="Continue", command=continue_eval)
        continue_button.pack(side=tk.LEFT, padx=5)

        # Focus window and bring to front
        root.focus_set()
        root.lift()

        # Center window (only on first creation)
        if not self._hitl_window_centered:
            root.update_idletasks()
            width = root.winfo_width()
            height = root.winfo_height()
            x = (root.winfo_screenwidth() // 2) - (width // 2)
            y = (root.winfo_screenheight() // 2) - (height // 2)
            root.geometry(f"{width}x{height}+{x}+{y}")
            self._hitl_window_centered = True

        # Wait for Continue button to be clicked
        root.wait_variable(continue_clicked)
        root.update()  # Process any pending events

        return edited_output, was_edited

    def close_hitl_window(self) -> None:
        """Close the HITL window if it exists."""
        if self._hitl_window:
            try:
                self._hitl_window.destroy()
            except Exception:
                pass
            self._hitl_window_centered = False
            self._hitl_window = None

    def create_hitl_evaluate_fn(
        self, lm: dspy.LM, metric: str = "exact"
    ) -> Callable[[Example, dict[str, str], str | None, str | None], float]:
        """Create a human-in-the-loop evaluation function.

        Args:
            lm: The DSPy language model to use for extraction.
            metric: Comparison metric to use. Options:
                - "exact-hitl": Score is 0 if edited, 1 if not edited
                - "levenshtein-hitl": Score is Levenshtein distance if edited, 1 if not edited

        Returns:
            An evaluation function that shows a GUI popup for human review.
        """
        # Track evaluation count for progress
        evaluation_counter = {"count": 0}
        total_examples = len(self.optimizer.examples) if hasattr(self.optimizer, "examples") else None

        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        def evaluate(
            example: Example,
            optimized_descriptions: dict[str, str],
            optimized_system_prompt: str | None,
            optimized_instruction_prompt: str | None,
        ) -> float:
            """HITL evaluation function that shows GUI popup.

            Args:
                example: The example with input_data and expected_output.
                optimized_descriptions: Dictionary of optimized field descriptions.
                optimized_system_prompt: Optimized system prompt (if provided).
                optimized_instruction_prompt: Optimized instruction prompt (if provided).

            Returns:
                Score between 0.0 and 1.0 based on user review.
            """
            # Build the extraction prompt (same as default evaluation)
            system_prompt = optimized_system_prompt or self.optimizer.system_prompt or ""
            instruction_prompt = optimized_instruction_prompt or self.optimizer.instruction_prompt or ""

            # Get input data from example
            input_data = example.input_data

            # Handle Pydantic models for input_data
            if isinstance(input_data, BaseModel):
                input_data = input_data.model_dump()

            # Extract text and images from input_data
            input_text: str | None = None
            images_raw: list[str] | list[Any] | None = None
            pil_images: list[Any] | None = None
            dspy_images: list[Any] | None = None

            if isinstance(input_data, dict):
                input_text = input_data.get("text")
                # Get images - prefer images_base64 if available (original base64)
                # Otherwise use images (which might be base64 strings or dspy.Image objects)
                images_raw = input_data.get("images_base64") or input_data.get("images")

                # Convert all images to PIL Image objects immediately
                if images_raw:
                    pil_images = self._convert_to_pil_images(images_raw)
                    
                    # Also keep base64 strings for DSPy conversion (if needed)
                    # Extract base64 strings from dspy.Image objects or use strings directly
                    images_for_dspy = []
                    if isinstance(images_raw, list) and len(images_raw) > 0:
                        first_item = images_raw[0]
                        if hasattr(first_item, "url"):
                            # These are dspy.Image objects, extract base64 from data URL
                            for img_obj in images_raw:
                                if hasattr(img_obj, "url"):
                                    url = img_obj.url
                                    if isinstance(url, str) and url.startswith("data:"):
                                        comma_idx = url.find(",")
                                        if comma_idx != -1:
                                            images_for_dspy.append(url[comma_idx + 1 :])
                                        else:
                                            images_for_dspy.append(url)
                                    elif hasattr(img_obj, "base64"):
                                        images_for_dspy.append(img_obj.base64)
                                    else:
                                        images_for_dspy.append(str(url))
                                else:
                                    images_for_dspy.append(str(img_obj))
                        elif isinstance(first_item, str):
                            # These are base64 strings
                            images_for_dspy = images_raw
                        else:
                            images_for_dspy = [str(img) for img in images_raw]
                    
                    # Convert base64 images to dspy.Image objects if present (for DSPy)
                    if images_for_dspy:
                        try:
                            dspy_images = convert_images_to_dspy_images(images_for_dspy)
                        except ImportError:
                            dspy_images = None
                
                # If no text but images exist, create a placeholder text
                if not input_text and pil_images:
                    input_text = "Extract structured data from the provided image(s)."
                elif not input_text:
                    input_text = str(input_data)
            else:
                input_text = str(input_data)

            # Apply optimized descriptions to the Pydantic model schema
            modified_schema = apply_optimized_descriptions(self.optimizer.model, optimized_descriptions)

            # Create the full prompt for extraction
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}")
            if instruction_prompt:
                prompt_parts.append(f"Instruction: {instruction_prompt}")

            prompt_parts.append(
                f"\nJSON Schema (with optimized field descriptions):\n"
                f"{json.dumps(modified_schema, indent=2)}"
            )

            if optimized_descriptions:
                prompt_parts.append("\nField descriptions summary:")
                for field_path, description in optimized_descriptions.items():
                    prompt_parts.append(f"  - {field_path}: {description}")

            if input_text:
                prompt_parts.append(f"\nInput text: {input_text}")
            if pil_images:
                prompt_parts.append(f"\nInput images: {len(pil_images)} image(s) provided")
            prompt_parts.append(
                "\nExtract the structured data according to the JSON schema above "
                "(which includes optimized field descriptions) and return it as valid JSON."
            )

            full_prompt = "\n\n".join(prompt_parts)
            json_prompt = f"{full_prompt}\n\nReturn only valid JSON, no other text."

            # For vision models, we need to pass images in the prompt
            # Note: We use dspy_images for the actual API call, not base64 strings
            if dspy_images:
                image_context = f"\n{len(dspy_images)} image(s) provided"
                json_prompt = f"{json_prompt}\n\n{image_context}"

            # Increment evaluation counter and show loading window before extraction
            evaluation_counter["count"] += 1
            current_eval_num = evaluation_counter["count"]

            # Show loading window before processing this evaluation
            self.show_loading_window(
                evaluation_num=current_eval_num,
                total_evaluations=total_examples,
            )

            # Use DSPy's ChainOfThought for extraction
            if dspy_images and len(dspy_images) > 0:
                if len(dspy_images) == 1:
                    signature = "prompt, image -> json_output"
                    extractor = dspy.ChainOfThought(signature)
                    result = extractor(prompt=json_prompt, image=dspy_images[0])
                else:
                    signature = "prompt, images -> json_output"
                    extractor = dspy.ChainOfThought(signature)
                    result = extractor(prompt=json_prompt, images=dspy_images)
            else:
                signature = "prompt -> json_output"
                extractor = dspy.ChainOfThought(signature)
                result = extractor(prompt=json_prompt)

            # Extract output text
            if hasattr(result, "json_output"):
                output_text = str(result.json_output)
            else:
                output_text = str(result)

            # Try to parse JSON directly
            extracted_data = None
            try:
                extracted_data = json.loads(output_text)
            except (json.JSONDecodeError, AttributeError):
                # Try to extract JSON from the text using regex
                json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
                json_match = re.search(json_pattern, output_text, re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                        if json_match:
                            try:
                                extracted_data = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                pass

            # If still no JSON found, use empty dict
            if extracted_data is None or not isinstance(extracted_data, dict):
                extracted_data = {}

            # Close loading window before showing popup
            self.close_loading_window()

            # Show HITL popup with progress info (reuses same window, updates content)
            # Pass PIL Image objects directly (they were converted earlier)
            edited_output, was_edited = self.show_hitl_popup(
                input_text=input_text,
                images=pil_images,  # Pass PIL Image objects directly
                proposed_output=extracted_data,
                evaluation_num=current_eval_num,
                total_evaluations=total_examples,
            )

            # Calculate score based on metric
            if metric == "exact-hitl":
                # Score is 0 if edited, 1 if not edited
                return 0.0 if was_edited else 1.0
            else:  # levenshtein-hitl
                # Score is Levenshtein distance if edited, 1 if not edited
                if was_edited:
                    original_str = json.dumps(extracted_data, sort_keys=True)
                    edited_str = json.dumps(edited_output, sort_keys=True)
                    max_len = max(len(original_str), len(edited_str))
                    if max_len == 0:
                        return 1.0
                    distance = levenshtein_distance(original_str, edited_str)
                    similarity = 1.0 - (distance / max_len)
                    return max(0.0, similarity)
                else:
                    return 1.0

        return evaluate

