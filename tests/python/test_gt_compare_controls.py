# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for requested versus published GT 1:1 state."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _HandleStub:
    def __init__(self):
        self.dirty_calls = []

    def dirty(self, name):
        self.dirty_calls.append(name)


class _ModelStub:
    def __init__(self):
        self.bound_funcs = {}
        self.bound_events = {}
        self.bound_binds = {}
        self.handle = _HandleStub()

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def bind(self, name, getter, setter):
        self.bound_binds[name] = (getter, setter)

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.classes = set()

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)


class _DocumentStub:
    def __init__(self):
        self.element = _ElementStub()

    def get_element_by_id(self, element_id):
        if element_id == "gt-compare-mode-block":
            return self.element
        return None


@pytest.fixture
def controller_environment(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        split_mode="gt_comparison",
        mode="rgb",
        requested=False,
        available=True,
        active=False,
        requested_writes=[],
    )

    def set_requested(enabled):
        state.requested = bool(enabled)
        state.requested_writes.append(state.requested)

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        get_split_view_mode=lambda: state.split_mode,
        get_gt_comparison_mode=lambda: state.mode,
        set_gt_comparison_mode=lambda mode: setattr(state, "mode", mode),
        get_gt_comparison_actual_size=lambda: state.requested,
        set_gt_comparison_actual_size=set_requested,
        is_gt_comparison_actual_size_available=lambda: state.available,
        is_gt_comparison_actual_size_active=lambda: state.active,
        tr=lambda key: key,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    ui_stub = ModuleType("lfs_plugins.ui")
    ui_stub.RuntimeState = SimpleNamespace(
        language_generation=SimpleNamespace(value=0),
    )
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_stub)

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.gt_compare_controls", None)
    module = import_module("lfs_plugins.gt_compare_controls")
    return module, state


@pytest.mark.parametrize(
    (
        "requested",
        "available",
        "active",
        "transitioning",
        "tooltip",
    ),
    (
        (
            True,
            True,
            False,
            True,
            "1:1 was requested, but a Fit preview is currently displayed while native-resolution content loads or recovers.",
        ),
        (
            True,
            True,
            True,
            False,
            "1:1 is active: one image pixel maps to one physical display pixel.",
        ),
        (
            True,
            False,
            False,
            False,
            "1:1 is unavailable for this camera or comparison mode. Distorted images require usable saved undistortion calibration.",
        ),
        (
            False,
            True,
            True,
            True,
            "1:1 is active: one image pixel maps to one physical display pixel.",
        ),
    ),
)
def test_controller_exposes_requested_available_active_and_transitioning(
    controller_environment,
    requested,
    available,
    active,
    transitioning,
    tooltip,
):
    module, state = controller_environment
    state.requested = requested
    state.available = available
    state.active = active

    controller = module.GTCompareControlsController()
    model = _ModelStub()
    document = _DocumentStub()
    controller.bind_model(model)
    controller.mount(document)
    assert controller.update(document) is not None

    assert model.bound_funcs["gt_compare_actual_size"]() is requested
    assert model.bound_funcs["gt_compare_actual_size_available"]() is available
    assert model.bound_funcs["gt_compare_actual_size_active"]() is active
    assert model.bound_funcs["gt_compare_actual_size_transitioning"]() is transitioning
    assert model.bound_funcs["gt_compare_actual_size_tooltip"]() == tooltip


def test_toggle_changes_requested_state_not_published_state(controller_environment):
    module, state = controller_environment
    controller = module.GTCompareControlsController()
    model = _ModelStub()
    document = _DocumentStub()
    controller.bind_model(model)
    controller.mount(document)
    controller.update(document)

    model.bound_events["gt_compare_toggle_actual_size"]()

    assert state.requested_writes == [True]
    assert state.active is False
    assert model.bound_funcs["gt_compare_actual_size"]() is True
    assert model.bound_funcs["gt_compare_actual_size_active"]() is False
    assert model.bound_funcs["gt_compare_actual_size_transitioning"]() is True


def test_rml_selection_uses_published_state_and_exposes_transition():
    project_root = Path(__file__).parent.parent.parent
    source = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "viewport_overlay.rml"
    ).read_text(encoding="utf-8")

    assert 'data-class-selected="gt_compare_actual_size_active"' in source
    assert 'data-class-transitioning="gt_compare_actual_size_transitioning"' in source
    assert 'data-attr-title="gt_compare_actual_size_tooltip"' in source
    assert 'data-class-selected="gt_compare_actual_size"' not in source
