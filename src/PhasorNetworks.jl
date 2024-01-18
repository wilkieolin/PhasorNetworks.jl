module PhasorNetworks

export 
#types
PhasorDense, 
PhasorODE,
SpikeTrain, 
MakeSpiking,
LocalCurrent,
SpikingArgs, 
SpikingCall, 
CurrentCall,

#spiking
default_spk_args,
phase_to_train,
phase_to_potential,
potential_to_phase,
train_to_phase,
count_nans,
zero_nans,
stack_trains,
vcat_trains,
delay_train,
match_offsets,

#vsa
bundle,
bundle_project,
bind,
unbind,
angle_to_complex,
chance_level,
complex_to_angle,
random_symbols,
similarity,
similarity_self,
similarity_outer,
similarity_loss,

#network
attend,
variance_scaling,

#metrics
cycle_correlation,
cor_realvals,
predict_quadrature,
accuracy_quadrature,
quadrature_loss,
similarity_loss,
loss_and_accuracy,
spiking_accuracy,
confusion_matrix,
OvR_matrices,
tpr_fpr,
interpolate_roc

include("metrics.jl")

end
