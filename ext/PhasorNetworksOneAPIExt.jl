module PhasorNetworksOneAPIExt

using PhasorNetworks
using oneAPI

# Add a Val{:oneapi} method to PhasorNetworks.select_device. The package
# defines select_device(::Symbol) → select_device(Val(backend)) plus
# methods for :cuda, :cpu, and a fallback. Extending via Val keeps each
# method's signature unique, so Julia's strict precompile (Aurora) does
# not flag this as method overwriting.
function PhasorNetworks.select_device(::Val{:oneapi})
    return oneAPI.oneAPIDevice()
end

end
