function classify_and_plot(adc_values, plotOutputPath, fileNameOnly)
    _7fff_values = UInt32[]
    _8000_values = UInt32[]
    other_values = UInt32[]

    for value in adc_values
        if value & 0xFFFF0000 == 0x7FFF0000
            push!(_7fff_values, value)
        elseif value & 0xFFFF0000 == 0x80000000
            push!(_8000_values, value)
        else
            push!(other_values, value)
        end
    end

    # Set x-axis range
    xmin = 2.147470E9
    xmax = 2.147490E9
    xmin = minimum(adc_values)
    xmax = maximum(adc_values)

    # Plot the first histogram with xlims
    p = histogram(_7fff_values,
        bins=250,
        label="Starts with 7FFF",
        title="Data Distribution",
        xlabel="Value",
        ylabel="Frequency",
        xlims=(xmin, xmax))

    # Add other histograms
    histogram!(p, _8000_values,
        bins=250,
        label="Starts with 8000")

    histogram!(p, other_values,
        bins=250,
        label="Other Values")

    # Save plot to file
    savefig(p, plotOutputPath * fileNameOnly * "/" * "classified_values_distribution.png")
end

function sample_data(times, voltages, interval)
    sampled_times = []
    sampled_voltages = []
    last_time = -interval  # Initialize to negative interval to ensure the first point is included

    for (t, v) in zip(times, voltages)
        if t - last_time >= interval
            push!(sampled_times, t)
            push!(sampled_voltages, v)
            last_time = t
        end
    end

    return sampled_times, sampled_voltages
end
