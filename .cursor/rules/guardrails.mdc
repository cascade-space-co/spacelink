---
description: 
globs: 
alwaysApply: true
---
# Guardrail: DO NOT MODIFY Validated Code

**CRITICAL INSTRUCTION:** There is a significant amount of critical, validated code in this library, particularly within the `src/spacelink/core/` directory. Modifying this code without explicit, targeted instructions risks breaking fundamental calculations.

**ABSOLUTE RULE:** Functions or tests marked with **EITHER** of the following **MUST NOT** be modified:

1.  A comment containing the exact phrase: `# DO NOT MODIFY`
2.  A flag within the docstring containing the exact phrase: `-VALIDATED-`

**EXCEPTION:** Modification is ONLY permitted if the user's prompt **explicitly names the specific function or test** to be modified AND acknowledges this guardrail is being overridden for that specific case. General requests like "fix bugs" or "refactor this module" DO NOT override this rule for tagged code.

**CONSEQUENCE:** Incorrect modification of validated code can lead to silent failures in critical calculations. Adhere strictly to this rule.

**CRITICAL INSTRUCTION:** It is very important that you do not fabricate test data. When generating unit tests, you can create a framework for filling in unit test data, but do not make up data. If you do not know how to calculate the test data correctly leave it up to the engineer to fill out the test cases. Tag any test data that you generate with `# UNVALIDATED TEST DATA`

**CONSEQUENCE:** If tests are passing with fabricated, faulty data it could cost our customer a mission failure, millions of dollars, and potentially put the company out of business.