module PhasorNetworks

export 
#types
PhasorDense, 
PhasorResonant,
SpikeTrain, 
SpikeTrainGPU,
MakeSpiking,
LocalCurrent,
SpikingArgs, 
SpikingCall, 
CurrentCall,
SpikingArgs_NN,

#domain conversions
phase_to_train,
phase_to_potential,
phase_to_current,
solution_to_potential,
solution_to_phase,
solution_to_train,
potential_to_phase,
train_to_phase,
time_to_phase,
phase_to_time,
potential_to_time,
time_to_potential,
arc_error,
cmpx_to_realvec,
realvec_to_cmpx,
period_to_angfreq,
angfreq_to_period,

#spiking
default_spk_args,
count_nans,
zero_nans,
stack_trains,
vcat_trains,
delay_train,
match_offsets,
mean_phase,
oscillator_bank,
neuron_constant,

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
dense_onehot,

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
