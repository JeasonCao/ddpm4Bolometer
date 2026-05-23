function read_uint32(io::IO, endianness)
    bytes = read(io, 4)  # Read 4 bytes
    if endianness == :LittleEndian 
        # If little-endian, Julia defaults to little-endian
        value = reinterpret(UInt32, bytes)[1]
    else
        # If big-endian, reverse byte order
        value = reinterpret(UInt32, reverse(bytes))[1]
    end
    return value
end

function read_float32(io::IO, endianness)
    bytes = read(io, 4) # Read 4 bytes
    if endianness == :LittleEndian 
        # If little-endian, Julia defaults to little-endian
        value = reinterpret(Float32, bytes)[1]
    else
        # If big-endian, reverse byte order
        value = reinterpret(Float32, reverse(bytes))[1]
    end
    return value
end

function read_header(io::IO)
    # Read the first 32-bit integer
    config = read(io, UInt32)
    
    # Extract Endianness and NBits from config
    endianness_char = Char((config >> 8) & 0xFF)
    nbits = config & 0xFF
    
    # Determine byte order
    is_little_endian = endianness_char == 'l'
    if is_little_endian
        endianness = :LittleEndian
    else
        endianness = :BigEndian
    end
    
    # Read sampling frequency (Float32)
    fs_uint = read_float32(io, endianness)
    fs = fs_uint  # Sampling frequency (Hz)
    
    # Read ADC full range (Float32)
    full_range = read_float32(io, endianness)
    
    return (endianness, nbits, fs, full_range)
end

function reverse_bits(value::UInt32)::UInt32
    binary_str = bitstring(value) # Get binary string representation
    reversed_str = reverse(binary_str) # Reverse the string
    reversed_value = parse(UInt32, reversed_str; base=2) # Parse the reversed string to value
    return reversed_value
end
