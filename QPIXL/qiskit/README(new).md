### `qpixl_module.py`
- Wraps an image or data source into a reusable quantum circuit using the QPIXL (cFRQI) encoding scheme.
- Supports compression, optional gate injection, and validation.
### `quantum_composer.py`

A flexible system to combine multiple quantum circuits using integration rules.

#### Key Features:

* Combines modules using rules like:
    * `sequential`: stack circuits in time
    * `merge`: align circuits side-by-side with optional entanglement
    * `hardware_aware_sequential`: transpile for specific coupling maps
* Built-in circuit safety (no duplicate bits)
* Modular `CircuitModule` interface for Qiskit circuits or QPIXL modules

#### Example:

```python
composer = QuantumComposer([qpixl_mod, algo_mod])
combined = composer.combine(rule="merge", connection_map={0: 5}, entangle_type="cx")
``````
### `QPIXL_demo_composer_extension.ipynb`

A complete test notebook demonstrating how to use the new modules.```

#### Covers:

* Basic image encoding + algorithm circuit
* Audio signal → QPIXL circuit integration
* Injecting gates during encoding
* Hardware-aware transpilation with `transpile()`
* Entanglement across subsystems
* Compression vs uncompressed QPIXL comparison
* `process_audio` helper 
    * You can define this in the notebook to convert `.mp3` audio signals into QPIXL-compatible angle arrays and circuits.
    * **Steps:**
        * Downsample audio → normalize → map to `[0, π]` rotation angles
        * Use `cFRQI()` to create circuit
        * Supports direct integration into `QPIXLModule`
