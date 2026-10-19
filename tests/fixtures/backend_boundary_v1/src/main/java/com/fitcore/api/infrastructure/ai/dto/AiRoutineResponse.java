package com.fitcore.api.infrastructure.ai.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.util.Collections;
import java.util.List;

import com.fitcore.api.domain.routine.dto.RoutineBlock;
import com.fitcore.api.infrastructure.ai.enums.GenerationStatus;
import com.fitcore.api.infrastructure.ai.enums.StatusReasonCode;

@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AiRoutineResponse {

    private String routineDraftId;

    private GenerationStatus generationStatus;

    private StatusReasonCode statusReasonCode;

    private Boolean isFallback;

    private Integer totalEstimatedTime;

    private String summaryTitle;

    private List<String> rationaleSummary;

    private List<RoutineBlock> routineBlocks;

    @Builder.Default
    private List<String> warnings = Collections.emptyList();
}
