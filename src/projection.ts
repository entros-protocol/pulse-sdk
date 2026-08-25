export type FeaturePipeline = "legacy" | "corrected" | "normalized-touch";

export type HyperplaneDefinition =
  | { family: "legacy" }
  | { family: "public"; transcriptVersion: 1 | 2 };

export interface ProjectionDefinition {
  featureSchemaVersion: 3 | 4 | 5;
  featurePipeline: FeaturePipeline;
  hyperplanes: HyperplaneDefinition;
  authenticatedTransitions: boolean;
}

const PROJECTION_DEFINITIONS = {
  0: {
    featureSchemaVersion: 3,
    featurePipeline: "legacy",
    hyperplanes: { family: "legacy" },
    authenticatedTransitions: false,
  },
  1: {
    featureSchemaVersion: 4,
    featurePipeline: "corrected",
    hyperplanes: { family: "public", transcriptVersion: 1 },
    authenticatedTransitions: true,
  },
  2: {
    featureSchemaVersion: 5,
    featurePipeline: "normalized-touch",
    hyperplanes: { family: "public", transcriptVersion: 2 },
    authenticatedTransitions: true,
  },
} as const satisfies Record<number, ProjectionDefinition>;

export const HIGHEST_SUPPORTED_PROJECTION_VERSION = 2;

export function getProjectionDefinition(
  projectionVersion: number,
): ProjectionDefinition {
  if (!Number.isInteger(projectionVersion)) {
    throw new Error(`Unsupported projection version ${projectionVersion}`);
  }

  const definition =
    PROJECTION_DEFINITIONS[
      projectionVersion as keyof typeof PROJECTION_DEFINITIONS
    ];
  if (!definition) {
    throw new Error(`Unsupported projection version ${projectionVersion}`);
  }
  return definition;
}
