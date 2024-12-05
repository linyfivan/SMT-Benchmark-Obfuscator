This is a tool to obfuscate smtlib2 benchmarks.
# Prerequisite
+ Require Linux and python packages in requirements.txt. (Not tested yet in other enviroments)
# Run
+ For arbitrary QF_SLIA/QF_LIA/QF_S smt2 file, first refer to https://github.com/SMT-COMP/scrambler and run scrambler once to obtain an anonymized benchmark. 
+ example.smt2 is a benchmark that is already anonymized if you just want to try out.
+ To run trans.py, type
```
python trans.py [input.smt2] [output.smt2]
```

for example, 
```
python trans.py example.smt2 result.smt2
```