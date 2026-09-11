/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <optional>
#include <shared_mutex>
#include <utility>

namespace lfs::vis {

    struct LiveModelLockBundle {
        std::shared_lock<std::shared_mutex> lock;
        const lfs::core::Scene* scene = nullptr;
        LiveModelLockBundle() = default;
        LiveModelLockBundle(std::shared_lock<std::shared_mutex>&& l, const lfs::core::Scene* s)
            : lock(std::move(l)),
              scene(s) {
            if (scene && lock.owns_lock()) {
                scene->noteLiveModelLockAcquired();
            }
        }
        LiveModelLockBundle(LiveModelLockBundle&& other) noexcept
            : lock(std::move(other.lock)),
              scene(other.scene) {
            other.scene = nullptr;
        }
        LiveModelLockBundle& operator=(LiveModelLockBundle&& other) noexcept {
            if (this != &other) {
                if (scene && lock.owns_lock()) {
                    scene->noteLiveModelLockReleased();
                }
                lock = std::move(other.lock);
                scene = other.scene;
                other.scene = nullptr;
            }
            return *this;
        }
        ~LiveModelLockBundle() {
            if (scene && lock.owns_lock()) {
                scene->noteLiveModelLockReleased();
            }
        }
        LiveModelLockBundle(const LiveModelLockBundle&) = delete;
        LiveModelLockBundle& operator=(const LiveModelLockBundle&) = delete;
        [[nodiscard]] bool owns_lock() const { return lock.owns_lock(); }
    };

    [[nodiscard]] inline std::optional<LiveModelLockBundle> acquireLiveModelRenderLock(
        const SceneManager* const scene_manager,
        const bool try_lock = false) {
        if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                const lfs::core::Scene* scene = trainer->getScene();
                if (try_lock) {
                    std::shared_lock<std::shared_mutex> candidate(
                        trainer->getRenderMutex(), std::try_to_lock);
                    if (!candidate.owns_lock()) {
                        return std::nullopt;
                    }
                    return LiveModelLockBundle(std::move(candidate), scene);
                }
                return LiveModelLockBundle(
                    std::shared_lock<std::shared_mutex>(trainer->getRenderMutex()), scene);
            }
        }
        return std::nullopt;
    }

} // namespace lfs::vis
