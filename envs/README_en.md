# README — Environment Correspondence
The two stacks correspond to the cross-environment statement in Appendix E of the paper:
the seven configurations with previously reported values were re-run on the workstation
stack and reproduced cell-by-cell, and three of the four newly added configurations also
ran on this stack; Mixtral-8x7B ran on the server stack, with the intervention engine
validated by the weight-space equivalence self-test (max deviation 2.4e-4).
Do not mix stacks when reproducing a given result.