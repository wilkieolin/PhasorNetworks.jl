using PhasorNetworks
using Documenter

DocMeta.setdocmeta!(PhasorNetworks, :DocTestSetup, :(using PhasorNetworks); recursive=true)

makedocs(;
    modules=[PhasorNetworks],
    authors="Wilkie Olin-Ammentorp",
    sitename="PhasorNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://wilkieolin.github.io/PhasorNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Types" => "api/types.md",
            "Networks" => "api/network.md",
            "Spiking" => "api/spiking.md",
            "Domains" => "api/domains.md",
            "VSA" => "api/vsa.md",
            "GPU" => "api/gpu.md",
            "Metrics" => "api/metrics.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/wilkieolin/PhasorNetworks.jl",
    devbranch="main",
)
