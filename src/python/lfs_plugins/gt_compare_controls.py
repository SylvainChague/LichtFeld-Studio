# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth comparison controls for the viewport overlay."""

import lichtfeld as lf

from .ui import RuntimeState


_DEFAULT_MODE = "rgb"


def _ui_label(key: str, fallback: str) -> str:
    tr = getattr(lf.ui, "tr", None)
    if not callable(tr):
        return fallback
    try:
        value = tr(key)
    except Exception:
        return fallback
    if value and value != key:
        return value
    return fallback


def _normalize_mode(value):
    value = str(value or "").strip().lower()
    if value in {"normal", "normals"}:
        return "normal"
    if value == "depth":
        return "depth"
    return _DEFAULT_MODE


class GTCompareControlsController:
    _DIRTY_FIELDS = (
        "gt_compare_tool_label",
        "gt_compare_mode_value",
        "gt_compare_actual_size",
        "gt_compare_actual_size_available",
        "gt_compare_actual_size_active",
        "gt_compare_actual_size_transitioning",
        "gt_compare_actual_size_tooltip",
    )

    def __init__(self):
        self._handle = None
        self._visible = False
        self._mode = _DEFAULT_MODE
        self._actual_size_requested = False
        self._actual_size_available = False
        self._actual_size_active = False
        self._actual_size_transitioning = False
        self._last_state_key = None

    @property
    def visible(self):
        return self._visible

    def bind_model(self, model):
        model.bind_func("gt_compare_tool_label", lambda: _ui_label("status_bar.gt_compare", "GT Compare"))
        model.bind(
            "gt_compare_mode_value",
            lambda: self._mode,
            self._set_mode,
        )
        model.bind_func(
            "gt_compare_actual_size",
            lambda: self._actual_size_requested,
        )
        model.bind_func(
            "gt_compare_actual_size_available",
            lambda: self._actual_size_available,
        )
        model.bind_func(
            "gt_compare_actual_size_active",
            lambda: self._actual_size_active,
        )
        model.bind_func(
            "gt_compare_actual_size_transitioning",
            lambda: self._actual_size_transitioning,
        )
        model.bind_func(
            "gt_compare_actual_size_tooltip",
            self._actual_size_tooltip,
        )
        model.bind_event(
            "gt_compare_toggle_actual_size",
            self._toggle_actual_size,
        )
        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None
        wrap = doc.get_element_by_id("gt-compare-mode-block")
        if wrap:
            wrap.set_class("hidden", True)

    def update(self, doc):
        dirty = False
        dirty_reasons = []
        visible = self._gt_compare_active()
        wrap = doc.get_element_by_id("gt-compare-mode-block")
        if wrap:
            wrap.set_class("hidden", not visible)

        if visible != self._visible:
            self._visible = visible
            dirty = True
            dirty_reasons.append("visibility")

        if not visible:
            self._last_state_key = None
            return ",".join(dirty_reasons) if dirty else None

        self._mode = self._read_mode()
        self._refresh_actual_size_state()
        state_key = (
            RuntimeState.language_generation.value,
            self._mode,
            self._actual_size_requested,
            self._actual_size_available,
            self._actual_size_active,
        )
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._dirty_all()
            dirty = True
            dirty_reasons.append("mode")
        return ",".join(dirty_reasons) if dirty else None

    def unmount(self):
        self._handle = None
        self._visible = False
        self._last_state_key = None

    def _gt_compare_active(self):
        getter = getattr(lf.ui, "get_split_view_mode", None)
        if not callable(getter):
            return False
        try:
            return getter() == "gt_comparison"
        except Exception:
            return False

    def _read_mode(self):
        getter = getattr(lf.ui, "get_gt_comparison_mode", None)
        if not callable(getter):
            return _DEFAULT_MODE
        try:
            return _normalize_mode(getter())
        except Exception:
            return _DEFAULT_MODE

    def _set_mode(self, value):
        mode = _normalize_mode(value)
        setter = getattr(lf.ui, "set_gt_comparison_mode", None)
        if callable(setter):
            try:
                setter(mode)
            except Exception:
                pass
            self._mode = self._read_mode()
        else:
            self._mode = _DEFAULT_MODE
        self._dirty_all()

    def _read_actual_size(self):
        getter = getattr(lf.ui, "get_gt_comparison_actual_size", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _read_actual_size_available(self):
        getter = getattr(lf.ui, "is_gt_comparison_actual_size_available", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _read_actual_size_active(self):
        getter = getattr(lf.ui, "is_gt_comparison_actual_size_active", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _refresh_actual_size_state(self):
        self._actual_size_requested = self._read_actual_size()
        self._actual_size_available = self._read_actual_size_available()
        self._actual_size_active = self._read_actual_size_active()
        self._actual_size_transitioning = (
            self._actual_size_available
            and self._actual_size_requested != self._actual_size_active
        )

    def _actual_size_tooltip(self):
        if not self._actual_size_available:
            return _ui_label(
                "tooltip.gt_compare_actual_size_unavailable",
                "1:1 is unavailable for this camera or comparison mode. Distorted images require usable saved undistortion calibration.",
            )
        if self._actual_size_requested and not self._actual_size_active:
            return _ui_label(
                "tooltip.gt_compare_actual_size_fallback",
                "1:1 was requested, but a Fit preview is currently displayed while native-resolution content loads or recovers.",
            )
        if self._actual_size_active:
            return _ui_label(
                "tooltip.gt_compare_actual_size_active",
                "1:1 is active: one image pixel maps to one physical display pixel.",
            )
        return _ui_label(
            "tooltip.gt_compare_actual_size",
            "Show RGB ground truth at one image pixel per physical display pixel. Requires a perspective camera and usable saved undistortion calibration for distorted images. Legacy projects may need dataset reimport and resave.",
        )

    def _toggle_actual_size(self, *_):
        if not self._read_actual_size_available():
            return
        setter = getattr(lf.ui, "set_gt_comparison_actual_size", None)
        if callable(setter):
            try:
                setter(not self._read_actual_size())
            except Exception:
                pass
        self._refresh_actual_size_state()
        self._dirty_all()

    def _dirty_all(self):
        if not self._handle:
            return
        for field in self._DIRTY_FIELDS:
            self._handle.dirty(field)
