# FACE3 smoke_timing summary

- status: failed
- experiment: edited_output_identity
- execution: sequential
- iterations per run: 2
- InstructPix2Pix edit steps in gradient loop: 20
- runs attempted: 4
- runs completed: 0
- failures: 4
- wall seconds: 34.55
- observed mean seconds/iteration/run: 0.000
- estimated 150-iteration full matrix: 2.3 min
- peak VRAM GB: 45.98936414718628
- all required per-iteration fields populated: True
- clamp/project logic active: True

## Failures

- edited_output_identity__face_002__add_black_sunglasses: OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 7.44 MiB is free. Process 3638029 has 8.03 MiB memory in use. Process 3638034 has 7.91 MiB memory in use. Process 3638025 has 7.91 MiB memory in use. Process 3638039 has 8.03 MiB memory in use. Including non-PyTorch memory, this process has 47.43 GiB memory in use. Of the allocated memory 45.97 GiB is allocated by PyTorch, and 1.11 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')
- edited_output_identity__face_002__add_headphones: OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 5.44 MiB is free. Process 3638029 has 8.03 MiB memory in use. Process 3638034 has 7.91 MiB memory in use. Process 3638025 has 7.91 MiB memory in use. Process 3638039 has 8.03 MiB memory in use. Including non-PyTorch memory, this process has 47.44 GiB memory in use. Of the allocated memory 45.99 GiB is allocated by PyTorch, and 1.10 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')
- edited_output_identity__face_005__add_black_sunglasses: OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 5.44 MiB is free. Process 3638029 has 8.03 MiB memory in use. Process 3638034 has 7.91 MiB memory in use. Process 3638025 has 7.91 MiB memory in use. Process 3638039 has 8.03 MiB memory in use. Including non-PyTorch memory, this process has 47.44 GiB memory in use. Of the allocated memory 45.99 GiB is allocated by PyTorch, and 1.10 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')
- edited_output_identity__face_005__add_headphones: OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 5.44 MiB is free. Process 3638029 has 8.03 MiB memory in use. Process 3638034 has 7.91 MiB memory in use. Process 3638025 has 7.91 MiB memory in use. Process 3638039 has 8.03 MiB memory in use. Including non-PyTorch memory, this process has 47.44 GiB memory in use. Of the allocated memory 45.99 GiB is allocated by PyTorch, and 1.10 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')