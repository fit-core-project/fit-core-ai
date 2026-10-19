package com.fitcore.api.domain.routine.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class RoutineBlock {
    @Min(1)
    private int order;
    @NotBlank
    private String exerciseId;
    @NotBlank
    private String exerciseName;
    private String movementPattern;
    private List<String> primaryMuscles;
    private String equipmentType;
    private int defaultRestSec;
    @NotEmpty
    @Valid
    private List<Prescription> prescription;
    private String exerciseRationale;
    private List<SubstitutionCandidates> substitutionCandidates;
}
