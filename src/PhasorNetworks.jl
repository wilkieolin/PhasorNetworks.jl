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

#domain conversions
phase_to_train,
phase_to_potential,
solution_to_potential,
solution_to_phase,
potential_to_phase,
train_to_phase,
time_to_phase,
phase_to_time,
arc_error,

#spiking
default_spk_args,
count_nans,
zero_nans,
stack_trains,
vcat_trains,
delay_train,
match_offsets,

#vsa
v_bundle,
v_bundle_project,
v_bind,
v_unbind,
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
