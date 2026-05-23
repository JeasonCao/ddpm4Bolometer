using Statistics
using Plots

include("parseIetiHeader.jl")
include("parseIetiData.jl")
include("convertIetiData.jl")
include("debugPlot.jl")
include("easyTrigger.jl")

# fileNames = [
#     "000081_20231122T174123_008_000",
#     "000093_20231126T194418_008_000",
#     "000111_20231128T142138_008_000",
#     "000088_20231123T154032_008_000",
#     "000105_20231127T164340_008_000",
#     "000111_20231128T142138_008_001",
#     "000091_20231124T160236_008_000",
#     "000108_20231127T183154_008_000",
#     "000112_20231129T102128_008_000",
#     "000092_20231125T201958_008_000",
#     "000108_20231127T183154_008_001",
#     "000092_20231125T201958_008_001",
#     "000109_20231128T093342_008_000"
# ]

fileNames = [
    "000091_20231124T160236_009_000"
]

plotOutputPath = "/home/shihongfu/LUCE.jl/cryoRun202311/plot/"

# Process each file in fileNames
for fileNameOnly in fileNames
    filename = "/home/shihongfu/data/LUCE_data/cryoRun202311/" * fileNameOnly * ".bin"
    result = process_triggered_signals(filename, plotOutputPath, fileNameOnly, 1000000)

    if result !== nothing
        trigger_indices, times, voltages = result

        # Assume times and voltages are your data
        interval = 0.01  # Sample every 0.01 seconds
        sampled_times, sampled_voltages = sample_data(times, voltages, interval)

        # Plot sampled data
        p = plot(sampled_times, sampled_voltages, xlabel="Time (s)", ylabel="Voltage (V)", title="ADC")

        # Save plot to file
        savefig(p, plotOutputPath * fileNameOnly * "/" * "sampled_streamdata.png")
    end
end

exit()
