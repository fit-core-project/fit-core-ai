package com.fitcore.api.domain.routine.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.math.BigDecimal;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class Prescription {
    private int setIndex;
    private String setType;
    private int targetReps;
    private BigDecimal targetWeightKg;
    private int targetRir;
    private int targetRestSec;
}
