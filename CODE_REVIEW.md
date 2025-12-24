# Code Review: Evolution & Limb Development

This review focuses on how evolutionary mechanisms and limb growth are modeled in the simulation. The aim is to highlight strengths, edge cases, and opportunities to better align morphology, control, and selection pressure.

## Observations
- **Morphology grows only from the core anchor.** Growth rules accept an `anchor` string, but `_anchor_node_id` always resolves to the core, preventing branching limbs or sensors from existing appendages.【F:organism/growth.py†L42-L118】
- **Genome rules are static and lack heritable variation.** `Genome` contains fixed growth rules with no mutation or crossover paths, and the starter genome hard-codes angles and lengths for a tri-/quad-like shape.【F:organism/genome.py†L22-L93】
- **Neural adaptation is disconnected from morphology.** The growth module notes that brain adaptation is external, but `Brain.build_starter` assumes actuator IDs supplied at construction time and offers no helper for adding/removing motor neurons as limbs change.【F:organism/growth.py†L9-L12】【F:neural/brain.py†L95-L164】
- **Selection and reproduction focus only on brains.** `Individual` wraps a brain and fitness, `next_generation` clones/mutates only brains, and there is no explicit link to genomes or morphologies, so limb evolution is indirect at best.【F:evolution/selection.py†L12-L19】【F:evolution/reproduction.py†L15-L38】【F:evolution/mutate.py†L11-L25】
- **Growth scheduling is deterministic and may bias rule usage.** `try_apply_growth` hashes `(org.age, _)` and tries up to six times, which can repeatedly pick the same rule order and starve others, especially with cooldowns.【F:organism/growth.py†L83-L123】
- **Energy use is global but not limb-aware.** Growth, kinematics, and neural control all draw from `Organism.energy`, yet per-node energy is unused beyond a field on `Node`, so limb upkeep/decay is not modeled.【F:organism/organism.py†L15-L58】【F:organism/nodes.py†L14-L41】

## Recommendations
- **Enable limb branching and targeted anchoring.** Extend `_anchor_node_id` to support anchors like `"random_node"`, `"actuator"`, or `"leaf"`, and allow rules to reference the newly created node IDs so limbs can chain outward (e.g., bud sensors from distal actuators).【F:organism/growth.py†L42-L118】
- **Introduce heritable genome mutation.** Add mutation operators that jitter angles/lengths/radii, duplicate/prune rules, and occasionally add new `GrowOp` types. Pair this with crossover to exchange limb sub-programs between parents, enabling genuine limb evolution instead of static templates.【F:organism/genome.py†L22-L93】
- **Co-evolve brain topology with morphology changes.** Provide utilities to add/remove motor neurons when actuators appear/disappear, and to wire new sensory nodes into the network with default gains. Tie these helpers to growth events or genome mutation so controllers keep pace with limb changes.【F:organism/growth.py†L9-L12】【F:neural/brain.py†L95-L164】
- **Broaden selection to whole organisms.** Redefine `Individual` to bundle genome + brain (and perhaps a snapshot of morphology stats). Mutate both in `next_generation`, ensuring limb structure is actually under selection pressure rather than implicitly fixed.【F:evolution/selection.py†L12-L19】【F:evolution/reproduction.py†L15-L38】
- **Add stochastic, priority-aware growth scheduling.** Replace the hash-based pick with weighted random selection that favors rules not recently used and perhaps scales by local resource density. This reduces bias and encourages diverse limb layouts across organisms.【F:organism/growth.py†L83-L123】
- **Model limb maintenance costs.** Use the per-node `energy` fields and edge lengths to apply upkeep/drag costs per limb segment, and allow high-cost limbs to atrophy (rule-driven pruning) when energy is low. This encourages evolution toward efficient appendage designs.【F:organism/organism.py†L15-L58】【F:organism/nodes.py†L14-L41】
- **Expose limb symmetry and modularity primitives.** Provide genome macros for mirrored growth (e.g., spawn paired actuators at ±θ) and repeated segments, letting evolution toggle whole modules instead of micromanaging angles. This can speed up emergence of coordinated limbs.【F:organism/genome.py†L22-L93】

## Testing & Validation Ideas
- Run multi-generation experiments where mutation rates for genomes and brains are varied independently, and track metrics like limb count, symmetry index, and locomotion efficiency.
- Visualize growth rule usage frequency to detect scheduling bias and confirm new anchoring options get exercised.
- Add regression tests for genome mutation to ensure rule cloning, duplication, and pruning keep cooldown/cost parameters consistent.
