# Point--dendrite and BP--local-credit controls

Status: complete; 200 new fits and the frozen
same-seed BP/LocalCA reference arms passed the numerical and resource gates.

- Serial tree minus resource-identical grouped star at aligned D3:
  30.81 percentage points
  (30.41 to 31.26;
  10/10 positive seeds).
- D3 alignment interaction for serial minus star:
  31.24 points
  (30.65 to 31.77).
- Full BP minus soma-broadcast autograd at aligned D3:
  0.64 points
  (0.15 to 1.17).
- Soma-broadcast autograd minus shared-soma LocalCA at aligned D3, with the
  LocalCA optimizer matched:
  -0.06 points
  (-0.23 to 0.09).
- Soma-broadcast autograd minus path-transport LocalCA at aligned D3, with the
  LocalCA optimizer matched:
  -11.02 points
  (-11.66 to -10.41).
- Changing only soma-broadcast from the BP optimizer to the LocalCA optimizer
  cost 13.47 points
  (12.89 to 14.05); after
  exact path transport, full BP retained a 3.08
  point advantage (2.48 to
  3.79).
- Active- and total-parameter-matched point MLP minus serial D3 BP:
  6.79 and
  7.20 points, respectively.

Interpretation is conditional on the frozen calibrated hierarchical gain--load
task. A positive serial--star interaction supports serial divisive composition;
a null contrast assigns the result to grouped computation rather than serial
dendritic depth. The point MLPs test whether this structural resource is an
unconstrained expressivity advantage. The soma-broadcast comparisons separate
teaching-coordinate restriction from the remaining optimizer-matched LocalCA
update-rule gap; they do not turn autograd into a biological mechanism.
