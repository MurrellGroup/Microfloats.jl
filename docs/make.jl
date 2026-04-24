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
        "The Microfloat type" => "microfloat.md",
        "Predefined types" => "predefined.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Microfloats.jl",
    devbranch="main",
)
