using Microfloats
using Documenter

DocMeta.setdocmeta!(Microfloats, :DocTestSetup, :(using Microfloats); recursive=true)

makedocs(;
    modules=[Microfloats],
    authors="Anton Oresten <antonoresten@gmail.com> and contributors",
    sitename="Microfloats.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Microfloats.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Microfloat" => "microfloat.md",
        "Conversion" => "conversion.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Microfloats.jl",
    devbranch="main",
)
