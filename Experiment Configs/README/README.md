# Method Config Readmes

This folder gives method-specific config guidance for the cleaned
`UL_UPLINK_DOWNLINK_MONTE_CARLO` experiment setup.

Use these files when you want to understand one method from top to bottom
without mixing in parameters that only matter to other methods.

Available guides:

- `uplink_convergence.md`
  - Applies to the uplink convergence baseline.
  - In the current cleaned folder, `Convergence per epoch` is the main public
    entry point and `Convergence per sweep` is the legacy wrapper around the
    same config surface.

- `uplink_monte_carlo.md`
  - Applies to the uplink training + testing Monte Carlo method.

- `downlink_convergence.md`
  - Applies to the downlink convergence baseline.
  - This is the guide to read when choosing between `direct_precoder`,
    `per_user_nets`, and `bs_shared_net`.

- `downlink_monte_carlo.md`
  - Applies to the downlink training + testing Monte Carlo method.
  - This is also the guide to read when choosing between per-user nets and one
    shared BS precoder net.

How to use these readmes:

1. Start with the decision variables.
2. Then tune the numeric variables by theme.
3. Set the scenario structure.
4. Set the physical system variables last.

For the full cross-method parameter reference, also see:

- `..\PARAMETER_GUIDE.md`
- `..\README.md`
