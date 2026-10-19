package com.fitcore.api.domain.routine.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fitcore.api.domain.exercise.entity.ExerciseTierEntity;
import com.fitcore.api.domain.exercise.repository.ExerciseTierRepository;
import com.fitcore.api.domain.routine.dto.Doms;
import com.fitcore.api.domain.routine.dto.Prescription;
import com.fitcore.api.domain.routine.util.MuscleMapper;
import com.fitcore.api.domain.routine.dto.RoutineBlock;
import com.fitcore.api.domain.routine.entity.RoutineDraftEntity;
import com.fitcore.api.domain.routine.entity.RoutineFinalEntity;
import com.fitcore.api.domain.routine.repository.RoutineDraftRepository;
import com.fitcore.api.domain.routine.repository.RoutineFinalRepository;
import com.fitcore.api.domain.user.components.UserComponent;
import com.fitcore.api.domain.routine.request.RoutineFinalRequest;
import com.fitcore.api.domain.routine.request.RoutineGenerateRequest;
import com.fitcore.api.domain.routine.response.RoutineDraftResponse;
import com.fitcore.api.domain.routine.response.RoutineFinalResponse;
import com.fitcore.api.global.common.util.SecurityUtils;
import com.fitcore.api.global.error.ErrorCode;
import com.fitcore.api.global.error.exception.BusinessException;
import com.fitcore.api.infrastructure.ai.client.AiClient;
import com.fitcore.api.infrastructure.ai.dto.AiRoutineRequest;
import com.fitcore.api.infrastructure.ai.dto.AiRoutineResponse;
import com.fitcore.api.infrastructure.ai.enums.GenerationStatus;
import com.fitcore.api.infrastructure.ai.enums.StatusReasonCode;
import com.fitcore.api.infrastructure.ai.fallback.AiFailureClassifier;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class RoutineService {
    private final RoutineDraftRepository routineDraftRepository;
    private final RoutineFinalRepository routineFinalRepository;
    private final SecurityUtils securityUtils;
    private final ObjectMapper objectMapper;
    private final AiClient aiClient;
    private final UserComponent userComponent;
    private final ExerciseTierRepository exerciseTierRepository;

    @Transactional
    public RoutineDraftResponse generateRoutine(RoutineGenerateRequest request) {
        log.info("AI Routine Generate UserId: {}", securityUtils.getCurrentUserId());
        AiRoutineRequest aiRequest = mapToAiRequest(request);

        try {
            AiRoutineResponse res = aiClient.generateRoutine(aiRequest);
            res.setIsFallback(Boolean.TRUE.equals(res.getIsFallback()));
            log.info("AI routine generation completed status={} fallback={}", res.getGenerationStatus(), res.getIsFallback());

            RoutineDraftEntity successEntity = RoutineDraftEntity.builder()
                .userId(securityUtils.getCurrentUserId())
                .generationStatus(res.getGenerationStatus())
                .statusReasonCode(res.getStatusReasonCode())
                .targetSplitLabel(request.getTargetSplitLabel() != null && !request.getTargetSplitLabel().isBlank() ? request.getTargetSplitLabel() : "custom")
                .isFallback(res.getIsFallback())
                .requestPayloadSnapshot(objectMapper.convertValue(request, new TypeReference<>() {
                }))
                .responsePayloadSnapshot(objectMapper.convertValue(res, new TypeReference<>() {
                }))
                .adapterRequestSnapshot(objectMapper.convertValue(aiRequest, new TypeReference<>() {
                }))
                .rationaleSummary(res.getRationaleSummary())
                .build();

            return RoutineDraftResponse.fromEntity(routineDraftRepository.save(successEntity));
        } catch (Exception e) {
            StatusReasonCode reasonCode = resolveFallbackReasonCode(e);
            log.warn(
                "event=ai_fallback endpoint=generate-routine reason_category={} fallback_used=true",
                reasonCode.name()
            );
            AiRoutineResponse fallbackResponse = createAiFailResponse(request, reasonCode);
            RoutineDraftEntity aiFailEntity = RoutineDraftEntity.builder()
                .userId(securityUtils.getCurrentUserId())
                .generationStatus(fallbackResponse.getGenerationStatus())
                .statusReasonCode(fallbackResponse.getStatusReasonCode())
                .targetSplitLabel(request.getTargetSplitLabel() != null && !request.getTargetSplitLabel().isBlank() ? request.getTargetSplitLabel() : "custom")
                .isFallback(Boolean.TRUE.equals(fallbackResponse.getIsFallback()))
                .requestPayloadSnapshot(objectMapper.convertValue(request, new TypeReference<>() {
                }))
                .responsePayloadSnapshot(
                    objectMapper.convertValue(fallbackResponse, new TypeReference<>() { // fallback routine
                    }))
                .adapterRequestSnapshot(objectMapper.convertValue(aiRequest, new TypeReference<>() {
                }))
                .rationaleSummary(fallbackResponse.getRationaleSummary())
                .build();

            return RoutineDraftResponse.fromEntity(routineDraftRepository.save(aiFailEntity));
        }
    }


    private StatusReasonCode resolveFallbackReasonCode(Exception e) {
        return AiFailureClassifier.classify(e);
    }

    private AiRoutineResponse createAiFailResponse(RoutineGenerateRequest request, StatusReasonCode reasonCode) {
        List<RoutineBlock> blocks = buildRequestAwareFallbackBlocks(request, reasonCode);

        AiRoutineResponse response = new AiRoutineResponse();
        response.setRoutineBlocks(blocks);

        if (blocks.isEmpty()) {
            response.setGenerationStatus(GenerationStatus.failed);
            response.setStatusReasonCode(StatusReasonCode.emptyCandidate);
            response.setIsFallback(false);
            response.setTotalEstimatedTime(0);
            response.setSummaryTitle("루틴을 만들 수 없음");
            response.setRationaleSummary(List.of("선택한 운동 부위, 통증, DOMS, 사용 불가 장비 조건을 모두 적용한 결과 사용 가능한 운동 후보가 없습니다."));
            response.setWarnings(List.of("운동 부위 또는 장비 조건을 변경한 뒤 다시 생성해 주세요."));
            log.warn("event=ai_fallback_empty_candidate reason_category={} fallback_used=false", reasonCode.name());
        } else {
            response.setGenerationStatus(GenerationStatus.fallback);
            response.setStatusReasonCode(reasonCode);
            response.setIsFallback(true);
            response.setTotalEstimatedTime(estimateFallbackTime(blocks));
            response.setSummaryTitle("기본 운동 루틴");
            response.setRationaleSummary(List.of(
                "AI 서버가 현재 닫혀 있어 기본 루틴으로 대체했습니다.",
                "통증이 있거나 컨디션이 좋지 않다면 강도를 낮춰 진행하세요."
            ));
            response.setWarnings(List.of("AI 서버 연결 실패로 개인화 정확도가 제한됩니다."));
        }

        return response;
    }

    private List<RoutineBlock> buildRequestAwareFallbackBlocks(RoutineGenerateRequest request, StatusReasonCode reasonCode) {
        List<RoutineBlock> catalogBlocks = buildCatalogFallbackBlocks(request, reasonCode);
        if (!catalogBlocks.isEmpty()) {
            return catalogBlocks;
        }

        Map<String, FallbackExercise> templates = new LinkedHashMap<>();
        templates.put("push", new FallbackExercise(
            "47", "Push-up", "horizontalPush", List.of("chest", "triceps"), "BODYWEIGHT", 3, 10, 90));
        templates.put("pull", new FallbackExercise(
            "65", "Inverted Row", "horizontalPull", List.of("upper-back", "biceps"), "BODYWEIGHT", 3, 10, 90));
        templates.put("legs", new FallbackExercise(
            "118", "Bodyweight Squat", "squat", List.of("quadriceps", "gluteal"), "BODYWEIGHT", 3, 12, 75));
        templates.put("core", new FallbackExercise(
            "120", "Plank", "antiExtension", List.of("abs"), "BODYWEIGHT", 3, 30, 60));

        List<FallbackExercise> selected = new ArrayList<>();
        String split = request.getTargetSplitLabel() == null ? "" : request.getTargetSplitLabel().trim().toLowerCase();
        if (templates.containsKey(split)) {
            selected.add(templates.get(split));
        } else if (request.getTargetMuscles() != null && !request.getTargetMuscles().isEmpty()) {
            selected.addAll(templates.values().stream()
                .filter(ex -> ex.primaryMuscles().stream().anyMatch(request.getTargetMuscles()::contains))
                .toList());
        }
        if (selected.isEmpty()) {
            selected.add(templates.get("core"));
        }

        List<String> painAreas = request.getCurrentPainAreas() == null
            ? Collections.emptyList()
            : request.getCurrentPainAreas();

        List<FallbackExercise> safeExercises = selected.stream()
            .filter(ex -> ex.primaryMuscles().stream().noneMatch(painAreas::contains))
            .toList();
        if (safeExercises.isEmpty()) {
            return Collections.emptyList();
        }

        List<RoutineBlock> safeBlocks = new ArrayList<>();
        for (int i = 0; i < safeExercises.size(); i++) {
            safeBlocks.add(toRoutineBlock(safeExercises.get(i), i + 1));
        }
        return safeBlocks;
    }

    private List<RoutineBlock> buildCatalogFallbackBlocks(RoutineGenerateRequest request, StatusReasonCode reasonCode) {
        List<String> targetMuscles = resolveFallbackTargetMuscles(request);
        Set<String> painAreas = request.getCurrentPainAreas() == null
            ? Collections.emptySet()
            : Set.copyOf(request.getCurrentPainAreas());
        Set<String> unavailableEquipment = request.getUnavailableEquipment() == null
            ? Collections.emptySet()
            : Set.copyOf(request.getUnavailableEquipment());

        List<ExerciseTierEntity> selected = exerciseTierRepository.findAll().stream()
            .filter(ex -> isFallbackCandidate(ex, targetMuscles, painAreas, unavailableEquipment))
            .sorted(Comparator
                .comparingInt((ExerciseTierEntity ex) -> isBodyweight(ex) ? 0 : 1)
                .thenComparingInt(ex -> ex.getDifficultyTier() == null ? 99 : ex.getDifficultyTier().intValue())
                .thenComparing(ExerciseTierEntity::getId))
            .limit(2)
            .toList();

        List<RoutineBlock> blocks = new ArrayList<>();
        for (int i = 0; i < selected.size(); i++) {
            blocks.add(toRoutineBlock(toFallbackExercise(selected.get(i)), i + 1));
        }
        if (!blocks.isEmpty()) {
            log.warn(
                "AI fallback catalog selected reason={} targetMuscles={} painAreas={} unavailableEquipment={} exerciseIds={}",
                reasonCode.name(),
                targetMuscles,
                painAreas,
                unavailableEquipment,
                blocks.stream().map(RoutineBlock::getExerciseId).toList()
            );
        }
        return blocks;
    }

    private List<String> resolveFallbackTargetMuscles(RoutineGenerateRequest request) {
        if (request.getTargetMuscles() != null && !request.getTargetMuscles().isEmpty()) {
            return normalizeTargetMuscles(request.getTargetMuscles());
        }
        String split = request.getTargetSplitLabel() == null ? "" : request.getTargetSplitLabel().trim().toLowerCase();
        return switch (split) {
            case "push" -> List.of("chest", "triceps", "front-deltoids");
            case "pull" -> List.of("upper-back", "biceps");
            case "legs" -> List.of("quadriceps", "gluteal");
            case "core" -> List.of("abs");
            default -> Collections.emptyList();
        };
    }

    private boolean isFallbackCandidate(
        ExerciseTierEntity exercise,
        List<String> targetMuscles,
        Set<String> painAreas,
        Set<String> unavailableEquipment
    ) {
        if (exercise.getPrimaryMuscle() == null || exercise.getNameEn() == null) {
            return false;
        }
        if (!targetMuscles.isEmpty() && !targetMuscles.contains(exercise.getPrimaryMuscle())) {
            return false;
        }
        if (containsAny(exercise.getPrimaryMuscle(), painAreas)
            || containsAny(exercise.getSecondaryMuscle(), painAreas)
            || containsAny(exercise.getPainTriggers(), painAreas)) {
            return false;
        }
        if (containsAny(exercise.getEquipmentReq(), unavailableEquipment)) {
            return false;
        }
        return isBodyweight(exercise) && (exercise.getDifficultyTier() == null || exercise.getDifficultyTier() <= 3);
    }

    private boolean containsAny(String csv, Set<String> values) {
        if (csv == null || csv.isBlank() || values.isEmpty()) {
            return false;
        }
        for (String token : csv.split(",")) {
            if (values.contains(token.trim())) {
                return true;
            }
        }
        return false;
    }

    private boolean isBodyweight(ExerciseTierEntity exercise) {
        return exercise.getEquipmentReq() != null && exercise.getEquipmentReq().contains("BODYWEIGHT");
    }

    private FallbackExercise toFallbackExercise(ExerciseTierEntity exercise) {
        return new FallbackExercise(
            String.valueOf(exercise.getId()),
            exercise.getNameEn(),
            null,
            collectMuscles(exercise),
            exercise.getEquipmentReq(),
            "STATIC".equalsIgnoreCase(exercise.getMovementType()) ? 3 : 2,
            "STATIC".equalsIgnoreCase(exercise.getMovementType()) ? 30 : 10,
            75
        );
    }

    private List<String> collectMuscles(ExerciseTierEntity exercise) {
        List<String> muscles = new ArrayList<>();
        muscles.add(exercise.getPrimaryMuscle());
        if (exercise.getSecondaryMuscle() != null && !exercise.getSecondaryMuscle().isBlank()) {
            for (String secondary : exercise.getSecondaryMuscle().split(",")) {
                String muscle = secondary.trim();
                if (!muscle.isBlank() && !muscles.contains(muscle)) {
                    muscles.add(muscle);
                }
            }
        }
        return muscles;
    }

    private RoutineBlock toRoutineBlock(FallbackExercise exercise, int order) {
        RoutineBlock block = new RoutineBlock();
        block.setOrder(order);
        block.setExerciseId(exercise.exerciseId());
        block.setExerciseName(exercise.exerciseName());
        block.setMovementPattern(exercise.movementPattern());
        block.setPrimaryMuscles(exercise.primaryMuscles());
        block.setEquipmentType(exercise.equipmentType());
        block.setDefaultRestSec(exercise.restSec());
        block.setExerciseRationale("AI 응답 실패 시에도 부상 부위를 피하도록 구성한 저위험 대체 운동입니다.");
        block.setSubstitutionCandidates(new ArrayList<>());

        List<Prescription> prescriptions = new ArrayList<>();
        for (int i = 1; i <= exercise.sets(); i++) {
            Prescription p = new Prescription();
            p.setSetIndex(i);
            p.setSetType("working");
            p.setTargetReps(exercise.reps());
            p.setTargetWeightKg(null);
            p.setTargetRir(3);
            p.setTargetRestSec(exercise.restSec());
            prescriptions.add(p);
        }
        block.setPrescription(prescriptions);
        return block;
    }

    private int estimateFallbackTime(List<RoutineBlock> blocks) {
        int seconds = blocks.stream()
            .flatMap(block -> block.getPrescription().stream())
            .mapToInt(p -> 45 + p.getTargetRestSec())
            .sum();
        return Math.max(5, (int) Math.ceil(seconds / 60.0));
    }

    private record FallbackExercise(
        String exerciseId,
        String exerciseName,
        String movementPattern,
        List<String> primaryMuscles,
        String equipmentType,
        int sets,
        int reps,
        int restSec
    ) {
    }

    private AiRoutineRequest mapToAiRequest(RoutineGenerateRequest request) {
        var profile = userComponent.findById();
        List<String> preferred = profile
            .map(p -> p.getPreferredExerciseIds() != null ? p.getPreferredExerciseIds() : Collections.<String>emptyList())
            .orElse(Collections.emptyList());
        List<String> unpreferred = profile
            .map(p -> p.getUnpreferredExerciseIds() != null ? p.getUnpreferredExerciseIds() : Collections.<String>emptyList())
            .orElse(Collections.emptyList());

        return AiRoutineRequest.builder()
            .userId(securityUtils.getCurrentUserId())
            .targetSplitLabel(request.getTargetSplitLabel())
            .targetMuscles(normalizeTargetMuscles(request.getTargetMuscles()))
            .readinessLevel(request.getReadinessLevel())
            .timeAvailableMin(request.getTimeAvailableMin())
            .painAreas(request.getCurrentPainAreas() == null ? Collections.emptyList() :
                request.getCurrentPainAreas().stream()
                    .map(area -> Map.of("bodyPart", area))
                    .toList())
            .domsData(convertDomsToMap(request.getCurrentDoms()))
            .equipment(request.getUnavailableEquipment())
            .goal(request.getGoal())
            .userNote(request.getUserNote())
            .preferredExerciseIds(preferred)
            .unpreferredExerciseIds(unpreferred)
            .build();
    }

    private Map<String, Integer> convertDomsToMap(List<Doms> doms) {
        return MuscleMapper.convertDomsToMap(doms);
    }

    private List<String> normalizeTargetMuscles(List<String> targetMuscles) {
        return MuscleMapper.normalizeTargetMuscles(targetMuscles);
    }

    @Transactional
    public RoutineDraftResponse createManualDraft(String title) {
        String userId = securityUtils.getCurrentUserId();
        String resolvedTitle = (title != null && !title.isBlank()) ? title : "새 루틴";

        AiRoutineResponse manualResponse = new AiRoutineResponse();
        manualResponse.setSummaryTitle(resolvedTitle);
        manualResponse.setGenerationStatus(GenerationStatus.success);
        manualResponse.setStatusReasonCode(StatusReasonCode.none);
        manualResponse.setIsFallback(false);
        manualResponse.setTotalEstimatedTime(0);
        manualResponse.setRationaleSummary(new ArrayList<>());
        manualResponse.setRoutineBlocks(new ArrayList<>());
        manualResponse.setWarnings(new ArrayList<>());

        RoutineDraftEntity entity = RoutineDraftEntity.builder()
            .userId(userId)
            .generationStatus(GenerationStatus.success)
            .statusReasonCode(StatusReasonCode.none)
            .targetSplitLabel("custom")
            .isFallback(false)
            .requestPayloadSnapshot(new RoutineGenerateRequest())
            .responsePayloadSnapshot(manualResponse)
            .rationaleSummary(new ArrayList<>())
            .build();

        return RoutineDraftResponse.fromEntity(routineDraftRepository.save(entity));
    }

    // 2. 猷⑦떞 ?뺤젙 (Request -> Entity -> Response)
    @Transactional
    public RoutineFinalResponse finalizeRoutine(String routineDraftId, RoutineFinalRequest request) {
        String currentUserId = securityUtils.getCurrentUserId();

        // Draft 議고쉶 諛?沅뚰븳 寃利?
        RoutineDraftEntity draft = routineDraftRepository.findById(routineDraftId)
            .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND));

        if (!draft.getUserId().equals(currentUserId)) {
            throw new BusinessException(ErrorCode.ACCESS_DENIED);
        }

        // Final Entity ?앹꽦
        RoutineFinalEntity finalEntity = RoutineFinalEntity.builder()
            .routineDraft(draft)
            .userId(currentUserId)
            .targetWorkoutDate(request.getTargetWorkoutDate())
            .targetSplitLabel(draft.getTargetSplitLabel() != null && !draft.getTargetSplitLabel().isBlank() ? draft.getTargetSplitLabel() : "custom")
            .finalRoutinePayload(request.getFinalRoutinePayload())
            .acceptedWithoutEdits(request.getAcceptedWithoutEdits())
            .userEditSummary(request.getUserEditSummary())
            .savedAt(LocalDateTime.now())
            .build();

        RoutineFinalEntity savedFinal = routineFinalRepository.save(finalEntity);
        return RoutineFinalResponse.fromEntity(savedFinal);
    }

    // 3. ???뺤젙 猷⑦떞 ?섏씠吏?議고쉶 (Entity Page -> Response Page 蹂??
    public Page<RoutineFinalResponse> getMyFinalRoutines(Pageable pageable) {
        String userId = securityUtils.getCurrentUserId();

        // Entity ?섏씠吏?議고쉶 ??Response DTO濡?留ㅽ븨?섏뿬 諛섑솚
        return routineFinalRepository.findByUserId(userId, pageable)
            .map(RoutineFinalResponse::fromEntity);
    }

    public RoutineFinalResponse getFinalRoutine(String finalId) {
        String currentUserId = securityUtils.getCurrentUserId();

        // 1. ?곗씠??議고쉶
        RoutineFinalEntity finalEntity = routineFinalRepository.findById(finalId)
            .orElseThrow(() -> new BusinessException(ErrorCode.ROUTINE_NOT_FOUND));

        // 2. 沅뚰븳 泥댄겕 (蹂몄씤??猷⑦떞?몄? ?뺤씤)
        if (!finalEntity.getUserId().equals(currentUserId)) {
            throw new BusinessException(ErrorCode.ACCESS_DENIED);
        }

        // 3. DTO 蹂??諛섑솚
        return RoutineFinalResponse.fromEntity(finalEntity);
    }
}

