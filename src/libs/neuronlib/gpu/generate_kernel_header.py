#!/usr/bin/env python3
import sys
import os

if len(sys.argv) != 3:
    print("Usage: generate_kernel_header.py <input.cl> <output.h>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read the OpenCL kernel source
with open(input_file, 'r') as f:
    content = f.read()

# Escape the content for C string
escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n"\n"')

# Write the header file
with open(output_file, 'w') as f:
    f.write('#pragma once\n')
    f.write('const char* gpu_kernel_source = "' + escaped + '";\n')

print(f"Generated {output_file} from {input_file}")