# Attacks on Linux

# Steps:

install python 3 `sudo apt install python3`
install tensorflow `pip3 install tensorflow`
install numpy `pip3 install numpy`

# for `./injectPDF.py`:

first run: `python3 ../patch_weights.py` and control-c out.
then run: `mkdir PDF_weights && cp python3 ../../PDF/w* ./PDF_weights`

- Run as: `sudo python3 injection <relevant file>.py <victim PID>`
- `injectXOR_ptrace.py` and `injectXOR_noPtrace.py` are there to show the attacks work with and without ptrace
