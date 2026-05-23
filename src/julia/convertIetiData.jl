function adc_to_voltage(adc_values, full_range, nbits)
    max_code = 2^nbits - 1
    voltages = adc_values .* (full_range / max_code)
    return voltages
end

function process_adc_file(filename, plotOutputPath, fileNameOnly; max_samples::Int=nothing)
    # Open log file
    log_filename = plotOutputPath * fileNameOnly * "/" * fileNameOnly * ".log"
    open(log_filename, "w") do log_io
        println(log_io, "Processing file: $filename\n")

        open(filename, "r") do io
            # Read header information
            (endianness, nbits, fs, full_range) = read_header(io)
            println(log_io, "Endianness: ", endianness == :LittleEndian ? "Little Endian" : "Big Endian")
            println(log_io, "NBits: ", nbits)
            println(log_io, "Sampling Frequency (Hz): ", fs)
            println(log_io, "ADC Full Range (V): ", full_range, "\n")

            # Read data section, pass max_samples parameter
            adc_values = read_data(io, endianness, nbits, filename; max_samples=max_samples)
            classify_and_plot(adc_values, plotOutputPath, fileNameOnly)

            # Convert to voltage values
            voltages = adc_to_voltage(adc_values, full_range, nbits)

            # Generate time series
            dt = 1 / fs
            times = collect(0:dt:(length(voltages)-1)*dt)

            return times, voltages
        end
    end
end
