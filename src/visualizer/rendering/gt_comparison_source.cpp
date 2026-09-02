/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"

#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "gt_comparison_cache_utils.hpp"
#include "io/pipelined_image_loader.hpp"
#include "rendering/image_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> resizePreview(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const glm::ivec2 target_size) {
            if (!image || !image->is_valid() || image->ndim() != 3 ||
                target_size.x <= 0 || target_size.y <= 0) {
                return {};
            }
            const auto layout = lfs::rendering::detectImageLayout(*image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return {};
            }

            lfs::core::Tensor source = *image;
            if (source.dtype() == lfs::core::DataType::UInt8) {
                source = source.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (source.dtype() != lfs::core::DataType::Float32) {
                source = source.to(lfs::core::DataType::Float32);
            }
            if (layout == lfs::rendering::ImageLayout::HWC) {
                source = source.permute({2, 0, 1}).contiguous();
            }
            source = source.cpu().contiguous();

            const int source_channels = static_cast<int>(source.size(0));
            const int source_height = static_cast<int>(source.size(1));
            const int source_width = static_cast<int>(source.size(2));
            if (source_channels <= 0 || source_width <= 0 || source_height <= 0) {
                return {};
            }
            if (source_width == target_size.x && source_height == target_size.y &&
                source_channels >= 3) {
                return std::make_shared<lfs::core::Tensor>(std::move(source));
            }

            const int destination_width = target_size.x;
            const int destination_height = target_size.y;
            const std::size_t destination_pixels =
                static_cast<std::size_t>(destination_width) * destination_height;
            std::vector<float> output(3 * destination_pixels, 0.0f);
            const float* const source_data = source.ptr<float>();
            if (!source_data) {
                return {};
            }
            const auto sample = [&](const int channel, const int x, const int y) {
                const int bounded_channel = std::clamp(channel, 0, source_channels - 1);
                return source_data[
                    (static_cast<std::size_t>(bounded_channel) * source_height + y) *
                        source_width +
                    x];
            };

            const float scale_x = static_cast<float>(source_width) / destination_width;
            const float scale_y = static_cast<float>(source_height) / destination_height;
            for (int y = 0; y < destination_height; ++y) {
                const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                int y0 = static_cast<int>(std::floor(source_y));
                float wy = source_y - static_cast<float>(y0);
                if (y0 < 0) {
                    y0 = 0;
                    wy = 0.0f;
                }
                int y1 = y0 + 1;
                if (y1 >= source_height) {
                    y1 = y0 = source_height - 1;
                    wy = 0.0f;
                }
                for (int x = 0; x < destination_width; ++x) {
                    const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    int x0 = static_cast<int>(std::floor(source_x));
                    float wx = source_x - static_cast<float>(x0);
                    if (x0 < 0) {
                        x0 = 0;
                        wx = 0.0f;
                    }
                    int x1 = x0 + 1;
                    if (x1 >= source_width) {
                        x1 = x0 = source_width - 1;
                        wx = 0.0f;
                    }
                    const std::size_t destination_index =
                        static_cast<std::size_t>(y) * destination_width + x;
                    for (int channel = 0; channel < 3; ++channel) {
                        const float top = std::lerp(
                            sample(channel, x0, y0), sample(channel, x1, y0), wx);
                        const float bottom = std::lerp(
                            sample(channel, x0, y1), sample(channel, x1, y1), wx);
                        output[static_cast<std::size_t>(channel) * destination_pixels +
                               destination_index] = std::lerp(top, bottom, wy);
                    }
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {3, static_cast<std::size_t>(destination_height),
                 static_cast<std::size_t>(destination_width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }
    } // namespace

    bool RenderingManager::gtRequestMatches(
        const GTComparisonPreviewRequest& lhs,
        const GTComparisonPreviewRequest& rhs) {
        return lhs.camera_uid == rhs.camera_uid &&
               lhs.mode == rhs.mode &&
               lhs.image_path == rhs.image_path &&
               lhs.image_size == rhs.image_size &&
               lhs.undistort_requested == rhs.undistort_requested &&
               lhs.depth_visualization_mode == rhs.depth_visualization_mode &&
               lhs.background_color == rhs.background_color;
    }

    bool RenderingManager::gtCacheEntryMatches(
        const GTComparisonImageCacheEntry& entry,
        const GTComparisonPreviewRequest& request) {
        return entry.camera_uid == request.camera_uid &&
               entry.mode == request.mode &&
               entry.image_path == request.image_path &&
               entry.image_size == request.image_size &&
               entry.undistort_requested == request.undistort_requested &&
               entry.depth_visualization_mode == request.depth_visualization_mode &&
               entry.background_color == request.background_color;
    }

    void RenderingManager::insertGTComparisonImageCacheEntry(
        const GTComparisonPreviewRequest& request,
        std::shared_ptr<lfs::core::Tensor> image,
        std::string error,
        const std::chrono::steady_clock::time_point now) {
        const bool image_valid = image && image->is_valid();
        assert(!image_valid || image->device() == lfs::core::Device::CPU);
        const std::size_t image_bytes = image_valid ? image->bytes() : 0;
        auto entry = std::find_if(
            gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
            [&request](const auto& candidate) {
                return gtCacheEntryMatches(candidate, request);
            });
        if (entry != gt_comparison_image_cache_.end()) {
            gt_comparison_image_cache_bytes_ -=
                entry->image && entry->image->is_valid() ? entry->image->bytes() : 0;
            *entry = {
                .camera_uid = request.camera_uid,
                .mode = request.mode,
                .undistort_requested = request.undistort_requested,
                .image_path = request.image_path,
                .image_size = request.image_size,
                .depth_visualization_mode = request.depth_visualization_mode,
                .background_color = request.background_color,
                .image = std::move(image),
                .error = std::move(error),
                .failure_time = image_valid ? std::chrono::steady_clock::time_point{} : now,
                .last_used = now};
        } else {
            gt_comparison_image_cache_.push_back({.camera_uid = request.camera_uid,
                                                  .mode = request.mode,
                                                  .undistort_requested = request.undistort_requested,
                                                  .image_path = request.image_path,
                                                  .image_size = request.image_size,
                                                  .depth_visualization_mode = request.depth_visualization_mode,
                                                  .background_color = request.background_color,
                                                  .image = std::move(image),
                                                  .error = std::move(error),
                                                  .failure_time = image_valid ? std::chrono::steady_clock::time_point{} : now,
                                                  .last_used = now});
        }
        gt_comparison_image_cache_bytes_ += image_bytes;

        while (gt_comparison_image_cache_.size() > GT_COMPARISON_IMAGE_CACHE_MAX_ENTRIES ||
               gt_comparison_image_cache_bytes_ > GT_COMPARISON_IMAGE_CACHE_MAX_BYTES) {
            const auto lru = std::min_element(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.last_used < rhs.last_used;
                });
            if (lru == gt_comparison_image_cache_.end()) {
                break;
            }
            gt_comparison_image_cache_bytes_ -=
                lru->image && lru->image->is_valid() ? lru->image->bytes() : 0;
            gt_comparison_image_cache_.erase(lru);
        }
    }

    RenderingManager::GTComparisonImageLookup
    RenderingManager::getOrQueueGTComparisonImage(
        GTComparisonPreviewRequest request) {
        GTComparisonImageLookup result;
        bool queued = false;
        const auto now = std::chrono::steady_clock::now();
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = request.camera_uid,
            .image_path = request.image_path};
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            auto cache_entry = std::find_if(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&request](const auto& entry) {
                    return gtCacheEntryMatches(entry, request);
                });
            if (cache_entry != gt_comparison_image_cache_.end()) {
                cache_entry->last_used = now;
                const bool failed =
                    !cache_entry->image || !cache_entry->image->is_valid();
                const bool retry_ready =
                    failed &&
                    cache_entry->failure_time.time_since_epoch().count() != 0 &&
                    now - cache_entry->failure_time >=
                        GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                if (!retry_ready) {
                    result.image = cache_entry->image;
                    result.error = cache_entry->error;
                    result.status = failed ? GTComparisonImageStatus::Failed
                                           : GTComparisonImageStatus::Ready;
                    return result;
                }
                result.status = GTComparisonImageStatus::Failed;
                result.error = cache_entry->error;
            }

            auto* active = active_gt_comparison_worker_request_
                               ? std::get_if<GTComparisonPreviewRequest>(
                                     &*active_gt_comparison_worker_request_)
                               : nullptr;
            const bool same_as_active = active && gtRequestMatches(*active, request);
            if (same_as_active &&
                (active_gt_comparison_image_is_prefetch_ ||
                 active->generation != gt_comparison_preview_request_generation_)) {
                active_gt_comparison_image_is_prefetch_ = false;
                active->generation = ++gt_comparison_preview_request_generation_;
                pending_gt_comparison_image_request_.reset();
            }
            const bool active_current =
                same_as_active && !active_gt_comparison_image_is_prefetch_ &&
                active->generation == gt_comparison_preview_request_generation_;
            const bool pending_same =
                pending_gt_comparison_image_request_ &&
                gtRequestMatches(*pending_gt_comparison_image_request_, request);

            if (!active_current && !pending_same) {
                const auto prefetched = std::find_if(
                    prefetch_gt_comparison_image_requests_.begin(),
                    prefetch_gt_comparison_image_requests_.end(),
                    [&request](const auto& candidate) {
                        return gtRequestMatches(candidate, request);
                    });
                if (prefetched != prefetch_gt_comparison_image_requests_.end()) {
                    request = std::move(*prefetched);
                    prefetch_gt_comparison_image_requests_.erase(prefetched);
                }
                request.generation = ++gt_comparison_preview_request_generation_;
                if (request.queued_at.time_since_epoch().count() == 0) {
                    request.queued_at = now;
                }
                pending_gt_comparison_image_request_ = request;
                queued = true;
            }

            if (result.status != GTComparisonImageStatus::Failed) {
                result.status = GTComparisonImageStatus::Loading;
            }
            const GTComparisonPreviewRequest* in_flight = nullptr;
            if (pending_gt_comparison_image_request_ &&
                gtRequestMatches(*pending_gt_comparison_image_request_, request)) {
                in_flight = &*pending_gt_comparison_image_request_;
            } else if (active && gtRequestMatches(*active, request)) {
                in_flight = active;
            }

            const auto stale = std::max_element(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&source_key](const auto& lhs, const auto& rhs) {
                    const auto rank = [&source_key](const auto& entry) {
                        if (!entry.image || !entry.image->is_valid()) {
                            return 0;
                        }
                        return entry.camera_uid == source_key.camera_uid &&
                                       entry.image_path == source_key.image_path
                                   ? 2
                                   : 1;
                    };
                    const int lhs_rank = rank(lhs);
                    const int rhs_rank = rank(rhs);
                    return lhs_rank != rhs_rank ? lhs_rank < rhs_rank
                                                : lhs.last_used < rhs.last_used;
                });
            if (stale != gt_comparison_image_cache_.end() && stale->image &&
                stale->image->is_valid()) {
                result.stale_image = stale->image;
            }
            result.grace_elapsed =
                !in_flight || in_flight->queued_at.time_since_epoch().count() == 0 ||
                now - in_flight->queued_at >= GT_COMPARISON_IMAGE_GRACE_PERIOD;
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
        return result;
    }

    RenderingManager::GTComparisonFullSourceLookup
    RenderingManager::getOrQueueGTComparisonFullSource(
        GTComparisonFullSourceRequest request) {
        GTComparisonFullSourceLookup result;
        std::optional<GTComparisonFullSourceSlot> released_source;
        bool queued = false;
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            if (gt_comparison_full_source_slot_ &&
                gt_comparison_full_source_slot_->source_key == request.source_key) {
                auto& slot = *gt_comparison_full_source_slot_;
                result.status = slot.status;
                result.generation = slot.generation;
                result.source = slot.cpu_source;
                result.error = slot.error;
                const bool retry_ready =
                    slot.status == GTComparisonImageStatus::Failed &&
                    slot.failure_time.time_since_epoch().count() != 0 &&
                    now - slot.failure_time >= GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                if (!retry_ready) {
                    return result;
                }
            }

            request.generation = ++gt_comparison_full_source_generation_;
            pending_gt_comparison_full_source_request_ = request;
            prefetch_gt_comparison_image_requests_.clear();
            released_source = std::move(gt_comparison_full_source_slot_);
            gt_comparison_full_source_slot_ = GTComparisonFullSourceSlot{
                .status = GTComparisonImageStatus::Loading,
                .source_key = request.source_key,
                .generation = request.generation};
            result.status = GTComparisonImageStatus::Loading;
            result.generation = request.generation;
            queued = true;
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
        return result;
    }

    void RenderingManager::queueGTComparisonImagePrefetch(
        GTComparisonPreviewRequest request) {
        bool queued = false;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            const bool in_cache = std::any_of(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&request](const auto& entry) {
                    return gtCacheEntryMatches(entry, request);
                });
            const auto* active = active_gt_comparison_worker_request_
                                     ? std::get_if<GTComparisonPreviewRequest>(
                                           &*active_gt_comparison_worker_request_)
                                     : nullptr;
            const bool already_queued =
                (pending_gt_comparison_image_request_ &&
                 gtRequestMatches(*pending_gt_comparison_image_request_, request)) ||
                (active && gtRequestMatches(*active, request)) ||
                std::any_of(
                    prefetch_gt_comparison_image_requests_.begin(),
                    prefetch_gt_comparison_image_requests_.end(),
                    [&request](const auto& candidate) {
                        return gtRequestMatches(candidate, request);
                    });
            if (!in_cache && !already_queued) {
                request.queued_at = std::chrono::steady_clock::now();
                if (prefetch_gt_comparison_image_requests_.size() >=
                    GT_COMPARISON_IMAGE_PREFETCH_MAX_ENTRIES) {
                    prefetch_gt_comparison_image_requests_.pop_front();
                }
                prefetch_gt_comparison_image_requests_.push_back(std::move(request));
                queued = true;
            }
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
    }

    void RenderingManager::invalidateGTComparisonImageCache() {
        std::optional<GTComparisonFullSourceSlot> released_full_source;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            ++gt_comparison_preview_request_generation_;
            ++gt_comparison_full_source_generation_;
            gt_comparison_image_cache_.clear();
            gt_comparison_image_cache_bytes_ = 0;
            pending_gt_comparison_image_request_.reset();
            pending_gt_comparison_full_source_request_.reset();
            prefetch_gt_comparison_image_requests_.clear();
            released_full_source = std::move(gt_comparison_full_source_slot_);
        }
        gt_comparison_actual_size_state_.reset();
        gt_comparison_cuda_image_.reset();
        gt_comparison_cuda_source_ = nullptr;
        gt_comparison_cuda_generation_ = 0;
        gt_comparison_cuda_camera_uid_ = -1;
        gt_comparison_cuda_size_ = {0, 0};
        gt_comparison_cuda_undistorted_ = false;
        split_left_source_ = nullptr;
        split_left_source_size_ = {0, 0};
        split_left_source_camera_uid_ = -1;
        split_left_source_undistorted_ = false;
        split_left_image_generation_ = 0;
        gt_comparison_loading_placeholder_.reset();
        gt_comparison_failed_placeholder_.reset();
    }

    void RenderingManager::invalidateGTComparisonActualSizeTile() {
        gt_comparison_actual_size_state_.invalidateTile();
        gt_comparison_cuda_image_.reset();
        gt_comparison_cuda_source_ = nullptr;
        gt_comparison_cuda_generation_ = 0;
        gt_comparison_cuda_camera_uid_ = -1;
        gt_comparison_cuda_size_ = {0, 0};
        gt_comparison_cuda_undistorted_ = false;
        split_left_source_ = nullptr;
        split_left_source_size_ = {0, 0};
        split_left_source_camera_uid_ = -1;
        split_left_source_undistorted_ = false;
    }

    void RenderingManager::invalidateGTComparisonActualSizeResources() {
        std::optional<GTComparisonFullSourceSlot> released_full_source;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            ++gt_comparison_full_source_generation_;
            pending_gt_comparison_full_source_request_.reset();
            prefetch_gt_comparison_image_requests_.clear();
            released_full_source = std::move(gt_comparison_full_source_slot_);
        }
        gt_comparison_actual_size_state_.reset();
        gt_comparison_cuda_image_.reset();
        gt_comparison_cuda_source_ = nullptr;
        gt_comparison_cuda_generation_ = 0;
        gt_comparison_cuda_camera_uid_ = -1;
        gt_comparison_cuda_size_ = {0, 0};
        gt_comparison_cuda_undistorted_ = false;
        split_left_source_ = nullptr;
        split_left_source_size_ = {0, 0};
        split_left_source_camera_uid_ = -1;
        split_left_source_undistorted_ = false;
        split_left_image_generation_ = 0;
    }

    RenderingManager::GTComparisonActualFrame
    RenderingManager::prepareGTActualFrame(
        const lfs::core::Camera& camera,
        const glm::ivec2 physical_viewport,
        cudaStream_t stream) {
        GTComparisonActualFrame frame;
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = camera.uid(),
            .image_path = camera.image_path()};

        if (gt_comparison_actual_size_state_.source_key != source_key) {
            gt_comparison_actual_size_state_.reset();
            gt_comparison_actual_size_state_.source_key = source_key;
            std::lock_guard lock(gt_comparison_image_mutex_);
            ++gt_comparison_preview_request_generation_;
            pending_gt_comparison_image_request_.reset();
            if (active_gt_comparison_image_is_prefetch_) {
                active_gt_comparison_image_is_prefetch_ = false;
            }
            prefetch_gt_comparison_image_requests_.clear();

            const auto fallback = std::max_element(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&source_key](const auto& lhs, const auto& rhs) {
                    const auto rank = [&source_key](const auto& entry) {
                        return entry.mode == GTComparisonMode::RGB &&
                                       entry.camera_uid == source_key.camera_uid &&
                                       entry.image_path == source_key.image_path &&
                                       entry.image && entry.image->is_valid()
                                   ? 1
                                   : 0;
                    };
                    const int lhs_rank = rank(lhs);
                    const int rhs_rank = rank(rhs);
                    return lhs_rank != rhs_rank ? lhs_rank < rhs_rank
                                                : lhs.last_used < rhs.last_used;
                });
            if (fallback != gt_comparison_image_cache_.end() &&
                fallback->mode == GTComparisonMode::RGB &&
                fallback->camera_uid == source_key.camera_uid &&
                fallback->image_path == source_key.image_path && fallback->image &&
                fallback->image->is_valid()) {
                gt_comparison_actual_size_state_.fit_fallback = fallback->image;
            }

            gt_comparison_image_cache_.remove_if(
                [&source_key](const GTComparisonImageCacheEntry& entry) {
                    return entry.camera_uid != source_key.camera_uid ||
                           entry.image_path != source_key.image_path;
                });
            gt_comparison_image_cache_bytes_ = 0;
            for (const auto& entry : gt_comparison_image_cache_) {
                if (entry.image && entry.image->is_valid()) {
                    gt_comparison_image_cache_bytes_ += entry.image->bytes();
                }
            }
        }

        const auto lookup = getOrQueueGTComparisonFullSource({.source_key = source_key});
        frame.status = lookup.status;
        frame.error = lookup.error;
        frame.fallback = gt_comparison_actual_size_state_.fit_fallback;
        if (lookup.status != GTComparisonImageStatus::Ready || !lookup.source ||
            !lookup.source->is_valid()) {
            return frame;
        }

        try {
            if (lookup.source->device() != lfs::core::Device::CPU ||
                lookup.source->dtype() != lfs::core::DataType::UInt8 ||
                lookup.source->ndim() != 3 || lookup.source->shape()[0] != 3 ||
                !lookup.source->is_contiguous()) {
                throw std::runtime_error(
                    "full-resolution source is not owned contiguous CPU CHW RGB8");
            }

            const glm::ivec2 source_extent{
                static_cast<int>(lookup.source->shape()[2]),
                static_cast<int>(lookup.source->shape()[1])};
            const bool distorted = camera.has_distortion();
            std::optional<lfs::core::UndistortParams> scaled_undistort;
            glm::ivec2 full_extent = source_extent;
            if (distorted) {
                scaled_undistort = lfs::core::scale_undistort_params(
                    camera.undistort_params(), source_extent.x, source_extent.y);
                full_extent = {
                    scaled_undistort->dst_width, scaled_undistort->dst_height};
            }

            const bool source_changed =
                gt_comparison_actual_size_state_.source_generation != lookup.generation;
            const bool recenter =
                source_changed ||
                gt_comparison_actual_size_state_.full_extent != full_extent ||
                gt_comparison_actual_size_state_.framebuffer_extent != physical_viewport;
            const auto crop = recenter
                                  ? detail::centerGTComparisonCrop(
                                        full_extent, physical_viewport)
                                  : detail::clampGTComparisonCrop(
                                        full_extent,
                                        physical_viewport,
                                        gt_comparison_actual_size_state_.crop.origin);
            if (!crop.valid()) {
                throw std::runtime_error("full-resolution comparison crop is empty");
            }

            if (source_changed) {
                gt_comparison_actual_size_state_.cpu_source = lookup.source;
                gt_comparison_actual_size_state_.cuda_source.reset();
                gt_comparison_actual_size_state_.source_generation = lookup.generation;
            }
            gt_comparison_actual_size_state_.full_extent = full_extent;
            gt_comparison_actual_size_state_.framebuffer_extent = physical_viewport;
            gt_comparison_actual_size_state_.crop = crop;

            const detail::GTComparisonTileKey tile_key{
                .source_generation = lookup.generation,
                .full_extent = full_extent,
                .framebuffer_extent = physical_viewport,
                .crop = crop,
                .distorted = distorted};
            if (!gt_comparison_actual_size_state_.tile_key ||
                *gt_comparison_actual_size_state_.tile_key != tile_key ||
                !gt_comparison_actual_size_state_.visible_tile) {
                lfs::core::Tensor visible;
                if (scaled_undistort) {
                    const lfs::core::CUDAStreamGuard guard(stream);
                    if (!gt_comparison_actual_size_state_.cuda_source) {
                        auto cuda_source =
                            lookup.source->to(lfs::core::Device::CUDA).contiguous();
                        gt_comparison_actual_size_state_.cuda_source =
                            std::make_shared<lfs::core::Tensor>(std::move(cuda_source));
                    }
                    visible = lfs::core::undistort_image_region(
                        *gt_comparison_actual_size_state_.cuda_source,
                        *scaled_undistort,
                        crop.origin.x,
                        crop.origin.y,
                        crop.extent.x,
                        crop.extent.y,
                        stream);
                    visible = lfs::rendering::flipImageVertical(
                        visible, lfs::rendering::ImageLayout::CHW);
                } else {
                    visible = lookup.source
                                  ->slice(
                                      1,
                                      static_cast<std::size_t>(crop.origin.y),
                                      static_cast<std::size_t>(
                                          crop.origin.y + crop.extent.y))
                                  .slice(
                                      2,
                                      static_cast<std::size_t>(crop.origin.x),
                                      static_cast<std::size_t>(
                                           crop.origin.x + crop.extent.x))
                                  .contiguous();
                    visible = lfs::rendering::flipImageVertical(
                        visible, lfs::rendering::ImageLayout::CHW);
                }
                gt_comparison_actual_size_state_.visible_tile =
                    std::make_shared<lfs::core::Tensor>(std::move(visible));
                gt_comparison_actual_size_state_.tile_key = tile_key;
            }

            frame.status = GTComparisonImageStatus::Ready;
            frame.tile = gt_comparison_actual_size_state_.visible_tile;
            frame.pixel_region = detail::GTComparisonPixelRegion{
                .origin = crop.origin,
                .full_extent = full_extent,
                .full_intrinsics =
                    scaled_undistort
                        ? std::optional{lfs::rendering::CameraIntrinsics{
                              .focal_x = scaled_undistort->dst_fx,
                              .focal_y = scaled_undistort->dst_fy,
                              .center_x = scaled_undistort->dst_cx,
                              .center_y = scaled_undistort->dst_cy}}
                        : std::nullopt};
            frame.content_rect = detail::centeredGTComparisonContentRect(
                physical_viewport, crop.extent);

            if (!gt_comparison_actual_size_state_.presented) {
                gt_comparison_actual_size_state_.fit_fallback.reset();
                frame.fallback.reset();
                std::lock_guard lock(gt_comparison_image_mutex_);
                gt_comparison_image_cache_.clear();
                gt_comparison_image_cache_bytes_ = 0;
                prefetch_gt_comparison_image_requests_.clear();
            }
            gt_comparison_actual_size_state_.presented = true;
        } catch (const std::exception& error) {
            gt_comparison_actual_size_state_.presented = false;
            gt_comparison_actual_size_state_.invalidateTile();
            frame.status = GTComparisonImageStatus::Failed;
            frame.tile.reset();
            frame.fallback.reset();
            frame.error = std::format(
                "RGB GT comparison 1:1 tile failed: {}", error.what());
            LOG_WARN("{}", frame.error);
        }
        return frame;
    }

    void RenderingManager::gtComparisonImageWorkerLoop(
        const std::stop_token stop_token) {
        if (const cudaError_t error = cudaStreamCreateWithFlags(
                &gt_comparison_worker_stream_, cudaStreamNonBlocking);
            error != cudaSuccess) {
            LOG_ERROR(
                "GT comparison worker stream creation failed ({}); GT image worker disabled",
                cudaGetErrorString(error));
            return;
        }
        const cudaStream_t worker_stream = gt_comparison_worker_stream_;
        std::optional<lfs::core::CUDAStreamGuard> worker_stream_guard;
        if (worker_stream) {
            worker_stream_guard.emplace(worker_stream);
        }
        const auto release_worker_stream = [this]() {
            if (gt_comparison_worker_stream_) {
                (void)cudaStreamSynchronize(gt_comparison_worker_stream_);
                lfs::core::CudaMemoryPool::instance().release_stream(
                    gt_comparison_worker_stream_);
                (void)cudaStreamDestroy(gt_comparison_worker_stream_);
                gt_comparison_worker_stream_ = nullptr;
            }
        };

        while (true) {
            GTComparisonWorkerRequest request;
            bool is_prefetch = false;
            {
                std::unique_lock lock(gt_comparison_image_mutex_);
                gt_comparison_image_cv_.wait(lock, stop_token, [this] {
                    return pending_gt_comparison_full_source_request_.has_value() ||
                           pending_gt_comparison_image_request_.has_value() ||
                           !prefetch_gt_comparison_image_requests_.empty();
                });
                if (stop_token.stop_requested()) {
                    pending_gt_comparison_full_source_request_.reset();
                    pending_gt_comparison_image_request_.reset();
                    prefetch_gt_comparison_image_requests_.clear();
                    active_gt_comparison_worker_request_.reset();
                    active_gt_comparison_image_is_prefetch_ = false;
                    worker_stream_guard.reset();
                    release_worker_stream();
                    return;
                }

                if (pending_gt_comparison_full_source_request_) {
                    request = std::move(*pending_gt_comparison_full_source_request_);
                    pending_gt_comparison_full_source_request_.reset();
                } else if (pending_gt_comparison_image_request_) {
                    request = std::move(*pending_gt_comparison_image_request_);
                    pending_gt_comparison_image_request_.reset();
                } else {
                    request = std::move(prefetch_gt_comparison_image_requests_.back());
                    prefetch_gt_comparison_image_requests_.pop_back();
                    is_prefetch = true;
                }
                active_gt_comparison_worker_request_ = request;
                active_gt_comparison_image_is_prefetch_ = is_prefetch;
            }

            std::shared_ptr<lfs::core::Tensor> image;
            std::string error;
            if (const auto* full =
                    std::get_if<GTComparisonFullSourceRequest>(&request)) {
                try {
                    auto source = lfs::core::load_image_rgb8_chw_lossless(
                        full->source_key.image_path);
                    image = std::make_shared<lfs::core::Tensor>(std::move(source));
                } catch (const std::exception& exception) {
                    error = std::format(
                        "RGB GT comparison full source load failed: {}",
                        exception.what());
                    LOG_WARN("{}", error);
                } catch (...) {
                    error =
                        "RGB GT comparison full source load failed with an unknown error";
                    LOG_WARN("{}", error);
                }
            } else {
                const auto& preview =
                    std::get<GTComparisonPreviewRequest>(request);
                try {
                    lfs::core::Tensor gt_tensor;
                    if (preview.mode == GTComparisonMode::RGB) {
                        if (preview.image_loader) {
                            // The training loader's byte cache is bounded and keyed by
                            // plain path (original bytes training reuses), so writing it
                            // is fine; skip_blob_cache below guards only CacheLoader's
                            // unbounded per-preview-size re-encode caches.
                            lfs::io::LoadParams params;
                            params.resize_factor = -1;
                            params.max_width = preview.preview_max_dimension;
                            params.cuda_stream = worker_stream;
                            params.output_uint8 = true;
                            gt_tensor = preview.image_loader->load_image_immediate(
                                preview.image_path, params);
                        } else {
                            gt_tensor = lfs::core::load_image_cached(
                                {.path = preview.image_path,
                                 .resize_factor = -1,
                                 .max_width = preview.preview_max_dimension,
                                 .stream = worker_stream,
                                 .output_uint8 = true,
                                 .skip_blob_cache = true});
                        }
                    }

                    if (preview.mode == GTComparisonMode::RGB) {
                        if (gt_tensor.is_valid() && gt_tensor.ndim() == 3) {
                            const auto layout =
                                lfs::rendering::detectImageLayout(gt_tensor);
                            if (layout != lfs::rendering::ImageLayout::Unknown) {
                                const bool undistort =
                                    layout == lfs::rendering::ImageLayout::CHW &&
                                    preview.undistort_requested;
                                if (undistort) {
                                    if (gt_tensor.device() !=
                                        lfs::core::Device::CUDA) {
                                        gt_tensor = gt_tensor.to(
                                            lfs::core::Device::CUDA, worker_stream);
                                    }
                                    if (gt_tensor.dtype() ==
                                        lfs::core::DataType::UInt8) {
                                        gt_tensor = gt_tensor.to(
                                                        lfs::core::DataType::Float32) /
                                                    255.0f;
                                        if (worker_stream) {
                                            gt_tensor.set_stream(worker_stream);
                                        }
                                    }
                                    const auto scaled =
                                        lfs::core::scale_undistort_params(
                                            preview.undistort_params,
                                            lfs::rendering::imageWidth(
                                                gt_tensor, layout),
                                            lfs::rendering::imageHeight(
                                                gt_tensor, layout));
                                    gt_tensor = lfs::core::undistort_image(
                                        gt_tensor, scaled, worker_stream);
                                }
                                gt_tensor = lfs::rendering::flipImageVertical(
                                    gt_tensor, layout);
                                // Static GT display images must be decoupled from the
                                // CUDA pool while training can recycle device buffers
                                // mid-frame.
                                gt_tensor = gt_tensor.cpu();
                                image = std::make_shared<lfs::core::Tensor>(
                                    std::move(gt_tensor));
                                image = resizePreview(image, preview.image_size);
                            }
                        }
                    } else if (preview.camera) {
                        if (preview.mode == GTComparisonMode::Depth) {
                            auto depth = preview.camera->load_and_get_depth(
                                -1, preview.preview_max_dimension);
                            if (depth.is_valid() && depth.ndim() == 2) {
                                if (preview.undistort_requested) {
                                    const auto scaled =
                                        lfs::core::scale_undistort_params(
                                            preview.undistort_params,
                                            static_cast<int>(depth.shape()[1]),
                                            static_cast<int>(depth.shape()[0]));
                                    depth = lfs::core::undistort_mask(
                                        depth, scaled, worker_stream);
                                }
                                image =
                                    gt_comparison_detail::makeDepthDisplayTensor(
                                        depth,
                                        preview.depth_visualization_mode,
                                        preview.background_color);
                                image = resizePreview(image, preview.image_size);
                                if (image) {
                                    auto flipped =
                                        lfs::rendering::flipImageVertical(
                                            *image,
                                            lfs::rendering::ImageLayout::CHW);
                                    image =
                                        std::make_shared<lfs::core::Tensor>(
                                            std::move(flipped));
                                }
                            }
                        } else {
                            auto normal = preview.camera->load_and_get_normal(
                                -1,
                                preview.preview_max_dimension,
                                lfs::core::Camera::NormalPriorDecode{});
                            if (normal.is_valid() && normal.ndim() == 3) {
                                const auto layout =
                                    lfs::rendering::detectImageLayout(normal);
                                if (preview.undistort_requested &&
                                    layout !=
                                        lfs::rendering::ImageLayout::Unknown) {
                                    const auto scaled =
                                        lfs::core::scale_undistort_params(
                                            preview.undistort_params,
                                            lfs::rendering::imageWidth(
                                                normal, layout),
                                            lfs::rendering::imageHeight(
                                                normal, layout));
                                    normal = lfs::core::undistort_image(
                                        normal, scaled, worker_stream);
                                }
                                image =
                                    gt_comparison_detail::makeNormalDisplayTensor(
                                        normal);
                                image = resizePreview(image, preview.image_size);
                                if (image) {
                                    auto flipped =
                                        lfs::rendering::flipImageVertical(
                                            *image,
                                            lfs::rendering::ImageLayout::CHW);
                                    image =
                                        std::make_shared<lfs::core::Tensor>(
                                            std::move(flipped));
                                }
                            }
                        }
                    }

                    if (worker_stream) {
                        if (const cudaError_t sync_error =
                                cudaStreamSynchronize(worker_stream);
                            sync_error != cudaSuccess) {
                            throw std::runtime_error(
                                std::string(
                                    "GT comparison worker CUDA sync failed: ") +
                                cudaGetErrorString(sync_error));
                        }
                    }
                    image =
                        gt_comparison_detail::convertDisplayTensorToUInt8(image);
                    if (!image || !image->is_valid()) {
                        image.reset();
                        error =
                            preview.mode == GTComparisonMode::RGB
                                ? "RGB GT comparison could not load the source image"
                            : preview.mode == GTComparisonMode::Depth
                                ? "Depth GT comparison could not load the camera depth map"
                                : "Normal GT comparison could not load the camera normal map";
                    }
                } catch (const std::exception& exception) {
                    image.reset();
                    error = std::format(
                        "RGB GT comparison image load failed: {}",
                        exception.what());
                    LOG_WARN("{}", error);
                } catch (...) {
                    image.reset();
                    error =
                        "RGB GT comparison image load failed with an unknown error";
                    LOG_WARN("{}", error);
                }
            }

            bool applied = false;
            {
                std::lock_guard lock(gt_comparison_image_mutex_);
                if (const auto* completed_full =
                        std::get_if<GTComparisonFullSourceRequest>(&request)) {
                    const auto* active_full =
                        active_gt_comparison_worker_request_
                            ? std::get_if<GTComparisonFullSourceRequest>(
                                  &*active_gt_comparison_worker_request_)
                            : nullptr;
                    const bool active_matches =
                        active_full &&
                        active_full->source_key == completed_full->source_key &&
                        active_full->generation == completed_full->generation;
                    const bool current =
                        active_matches &&
                        completed_full->generation ==
                            gt_comparison_full_source_generation_ &&
                        gt_comparison_full_source_slot_ &&
                        gt_comparison_full_source_slot_->source_key ==
                            completed_full->source_key &&
                        gt_comparison_full_source_slot_->generation ==
                            completed_full->generation;
                    if (!stop_token.stop_requested() && current) {
                        auto& slot = *gt_comparison_full_source_slot_;
                        slot.status =
                            image && image->is_valid()
                                ? GTComparisonImageStatus::Ready
                                : GTComparisonImageStatus::Failed;
                        slot.cpu_source = std::move(image);
                        slot.error = std::move(error);
                        slot.failure_time =
                            slot.status == GTComparisonImageStatus::Failed
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
                        applied = true;
                    }
                    if (active_matches) {
                        active_gt_comparison_worker_request_.reset();
                        active_gt_comparison_image_is_prefetch_ = false;
                    }
                } else {
                    const auto& completed_preview =
                        std::get<GTComparisonPreviewRequest>(request);
                    auto* active_preview =
                        active_gt_comparison_worker_request_
                            ? std::get_if<GTComparisonPreviewRequest>(
                                  &*active_gt_comparison_worker_request_)
                            : nullptr;
                    const bool active_matches =
                        active_preview &&
                        gtRequestMatches(*active_preview, completed_preview);
                    const bool completed_as_prefetch =
                        active_gt_comparison_image_is_prefetch_;
                    const bool current =
                        active_matches &&
                        (completed_as_prefetch ||
                         active_preview->generation ==
                             gt_comparison_preview_request_generation_);
                    if (!stop_token.stop_requested() && current) {
                        insertGTComparisonImageCacheEntry(
                            completed_preview,
                            std::move(image),
                            std::move(error),
                            std::chrono::steady_clock::now());
                        if (completed_as_prefetch &&
                            pending_gt_comparison_image_request_ &&
                            gtRequestMatches(
                                *pending_gt_comparison_image_request_,
                                completed_preview)) {
                            pending_gt_comparison_image_request_.reset();
                        }
                        applied = true;
                    }
                    if (active_matches) {
                        active_gt_comparison_worker_request_.reset();
                        active_gt_comparison_image_is_prefetch_ = false;
                    }
                }
            }

            if (applied) {
                markDirty(DirtyFlag::SPLIT_VIEW);
            }
        }
    }

} // namespace lfs::vis
