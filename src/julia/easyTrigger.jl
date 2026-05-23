# Calculate baseline RMS and mean
function calculate_baseline_metrics(voltages, baseline_period=1000)
    actual_period = min(baseline_period, length(voltages))  # Adjust the period to the actual length of voltages
    baseline_data = voltages[1:actual_period]  # Take the first `actual_period` points as baseline
    mean_val = mean(baseline_data)  # Calculate mean value
    rms = sqrt(mean((baseline_data .- mean_val) .^ 2))  # Calculate RMS
    return mean_val, rms
end

# Find the first ten trigger points that exceed the threshold, ensuring non-overlapping events
function find_trigger_points(voltages, threshold_upper, threshold_lower, fs, pre_trigger_sec, post_trigger_sec, num_triggers=10)
    trigger_indices = []
    marked_times = falses(length(voltages))  # Used to mark already triggered periods

    pre_trigger_samples = round(Int, pre_trigger_sec * fs)
    post_trigger_samples = round(Int, post_trigger_sec * fs)

    for i in eachindex(voltages)
        # Skip if the current index is within an already marked period
        if marked_times[i] || (voltages[i] > threshold_lower && voltages[i] < threshold_upper)
            continue
        end

        push!(trigger_indices, i)

        # Mark the triggered period
        start_index = max(1, i - pre_trigger_samples)
        end_index = min(length(voltages), i + post_trigger_samples)
        marked_times[start_index:end_index] .= true

        if length(trigger_indices) >= num_triggers
            break
        end
    end
    return trigger_indices
end

# Extract signal segment
function extract_signal_segment(times, voltages, trigger_index, pre_trigger_sec, post_trigger_sec, fs)
    pre_trigger_samples = round(Int, pre_trigger_sec * fs)
    post_trigger_samples = round(Int, post_trigger_sec * fs)

    start_index = max(1, trigger_index - pre_trigger_samples)
    end_index = min(length(voltages), trigger_index + post_trigger_samples)

    return @view(times[start_index:end_index]), @view(voltages[start_index:end_index])
end

function process_triggered_signals(filename, plotOutputPath, fileNameOnly, max_samples, threshold_multiplier=9, pre_trigger_sec=0.03, post_trigger_sec=0.07, num_triggers=50)
    times, voltages = process_adc_file(filename, plotOutputPath, fileNameOnly; max_samples=max_samples)

    # Open log file
    log_filename = plotOutputPath * fileNameOnly * "/" * fileNameOnly * ".log"
    open(log_filename, "a") do log_io
        # Calculate baseline mean and RMS
        mean_val, rms = calculate_baseline_metrics(voltages)
        threshold_upper = mean_val + threshold_multiplier * rms
        threshold_lower = mean_val - threshold_multiplier * rms
        println(log_io, "Baseline RMS: ", rms)
        println(log_io, "Baseline Mean: ", mean_val)
        println(log_io, "Threshold upper: ", threshold_upper)
        println(log_io, "Threshold lower: ", threshold_lower, "\n")

        fs = 1 / (times[2] - times[1])  # Calculate sampling frequency

        # Find trigger points
        trigger_indices = find_trigger_points(voltages, threshold_upper, threshold_lower, fs, pre_trigger_sec, post_trigger_sec, num_triggers)

        if isempty(trigger_indices)
            println(log_io, "No trigger points exceeding the threshold found.")
            return nothing
        end

        for (idx, trigger_index) in enumerate(trigger_indices)
            println(log_io, "Trigger point index: ", trigger_index)

            # Extract signal segment
            segment_times, segment_voltages = extract_signal_segment(times, voltages, trigger_index, pre_trigger_sec, post_trigger_sec, fs)

            # Plot signal segment
            p = plot(segment_times, segment_voltages, xlabel="Time (s)", ylabel="Voltage (V)", title="ADC")

            # Save plot to file
            savefig(p, plotOutputPath * fileNameOnly * "/" * "triggered_signal_$(idx).png")
        end

        return trigger_indices, times, voltages
    end
end
