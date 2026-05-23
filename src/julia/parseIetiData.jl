using Logging

function read_data(io::IO, endianness, nbits, filename; max_samples::Int=nothing)
    adc_values = UInt32[]
    sample_count = 0

    @info "Start reading data from " * filename * " ..."

    while !eof(io)
        if nbits == 32
            value = read_uint32(io, endianness)
            push!(adc_values, value)

            sample_count += 1
            # If max_samples is set and the specified number of samples have been read, exit the loop
            if max_samples !== nothing && sample_count >= max_samples
                break
            end
        else
            # Special handling for cases where nbits is not 32
            error("Unsupported nbits value: $nbits")
        end
    end
    return adc_values
end
