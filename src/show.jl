function Base.show(io::IO, x::T) where T <: Microfloat
    show_typeinfo = get(IOContext(io), :typeinfo, nothing) != T
    type = repr(T)
    show_typeinfo && print(io, type, "(")
    print(io, Float64(x))
    show_typeinfo && print(io, ")")
    return nothing
end
