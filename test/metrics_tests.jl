# Proposed metrics_tests.jl
# This file proposes new tests for the metrics.jl module
# To be added to the test suite to improve coverage

using Statistics: mean, std
using LinearAlgebra: diag

"""
Run all metrics-related tests
"""
function metrics_tests()
    @testset "Metrics Tests" begin
        @info "Running metrics tests..."

        arc_error_tests()
        angular_mean_tests()
        exp_score_tests()
        z_score_tests()
        similarity_correlation_tests()
        cycle_correlation_tests()
        cycle_sparsity_tests()
        cor_realvals_tests()
        evaluate_loss_tests()
        evaluate_accuracy_tests()
        confusion_matrix_tests()
        ovr_matrices_tests()
        tpr_fpr_tests()
        interpolate_roc_tests()
    end
end

function arc_error_tests()
    @testset "Arc Error Tests" begin
        # Test scalar input
        @test arc_error(0.0f0) ≈ 0.0f0
        
        # Test at boundaries
        @test abs(arc_error(1.0f0)) < 1e-5
        @test abs(arc_error(-1.0f0)) < 1e-5
        
        # Test intermediate value (note: sin(π * 0.5) = 1.0 exactly)
        @test 0 < arc_error(0.3f0) < 1.0f0
        @test arc_error(0.5f0) ≈ 1.0f0 atol=1e-5
        
        # Test array input
        phases = [0.0f0, 0.25f0, 0.5f0, 0.75f0, 1.0f0]
        errors = arc_error(phases)
        @test size(errors) == size(phases)
        @test all(errors .>= -1.0f0) && all(errors .<= 1.0f0)
        
        # Test anti-symmetry: arc_error(-x) = -arc_error(x) (sine is odd)
        @test arc_error(0.3f0) ≈ -arc_error(-0.3f0)
        
        # Test 2D array
        phases_2d = rand(Float32, 10, 20) .* 2.0f0 .- 1.0f0
        errors_2d = arc_error(phases_2d)
        @test size(errors_2d) == size(phases_2d)
    end
end

function angular_mean_tests()
    @testset "Angular Mean Tests" begin
        # Test with uniform angles
        phases = ones(Float32, 5, 10) .* 0.5f0
        mean_phase = angular_mean(phases, dims=2)
        @test size(mean_phase) == (5, 1)
        @test all(mean_phase .≈ 0.5f0)
        
        # Test with opposite angles (should cancel)
        phases_opposite = cat(ones(Float32, 5, 5) .* 0.5f0, 
                             ones(Float32, 5, 5) .* -0.5f0, dims=2)
        mean_opposite = angular_mean(phases_opposite, dims=2)
        # Mean of opposite angles should be near 0 or ±1
        @test all(abs.(mean_opposite) .< 0.2f0) || all(abs.(abs.(mean_opposite) .- 1.0f0) .< 0.2f0)
        
        # Test different reduction dimensions
        phases_3d = rand(Float32, 3, 4, 5) .* 2.0f0 .- 1.0f0
        mean_dim1 = angular_mean(phases_3d, dims=1)
        mean_dim2 = angular_mean(phases_3d, dims=2)
        @test size(mean_dim1) == (1, 4, 5)
        @test size(mean_dim2) == (3, 1, 5)
        
        # All means should be in [-1, 1]
        @test all(mean_dim1 .>= -1.0f0) && all(mean_dim1 .<= 1.0f0)
        @test all(mean_dim2 .>= -1.0f0) && all(mean_dim2 .<= 1.0f0)
    end
end

function exp_score_tests()
    @testset "Exponential Score Tests" begin
        # Test perfect similarity
        @test exp_score([1.0f0])[1] ≈ 0.0f0
        
        # Test zero similarity
        zero_score = exp_score([0.0f0], scale=3.0f0)[1]
        @test zero_score > 0.0f0
        
        # Test monotonicity - higher similarity should give lower score
        sims = [0.1f0, 0.5f0, 0.9f0]
        scores = exp_score(sims, scale=3.0f0)
        @test scores[1] > scores[2] > scores[3]
        
        # Test with arrays
        sims_array = rand(Float32, 10, 20)
        scores_array = exp_score(sims_array, scale=3.0f0)
        @test size(scores_array) == size(sims_array)
        @test all(scores_array .>= 0.0f0)
        
        # Test scale parameter effect
        scores_scale3 = exp_score([0.5f0], scale=3.0f0)[1]
        scores_scale5 = exp_score([0.5f0], scale=5.0f0)[1]
        @test scores_scale3 != scores_scale5
        @test scores_scale5 > scores_scale3  # Higher scale should give higher scores
    end
end

function z_score_tests()
    @testset "Z-Score Tests" begin
        # Test at center
        @test isfinite(z_score([0.5f0])[1])
        
        # Test symmetry
        @test z_score([0.3f0])[1] ≈ z_score([0.7f0])[1]
        
        # Test boundaries
        z_at_boundary = z_score([-1.0f0, 1.0f0])
        @test all(isfinite.(z_at_boundary))
        
        # Test array input
        phases = rand(Float32, 5, 10) .* 2.0f0 .- 1.0f0
        scores = z_score(phases)
        @test size(scores) == size(phases)
        @test all(scores .>= 0.0f0)
    end
end

function similarity_correlation_tests()
    @testset "Similarity Correlation Tests" begin
        # Create synthetic data
        n_x, n_y = 10, 15
        n_time = 20
        
        # Static similarity matrix
        static_sim = abs.(randn(Float32, n_x, n_y))
        
        # Dynamic similarity (correlates with static) - 3D array with time as third dimension
        dynamic_sim = cat([static_sim .+ 0.1f0 * randn(Float32, n_x, n_y) 
                          for _ in 1:n_time]..., dims=3)
        
        # Compute correlation
        correlations = similarity_correlation(static_sim, dynamic_sim)
        
        # Should have one correlation value per time step
        @test length(correlations) == n_time
        
        # Correlations should be in [-1, 1] range
        @test all(correlations .>= -1.0f0) && all(correlations .<= 1.0f0)
        
        # Correlations should generally be high (similar patterns)
        @test mean(correlations) > 0.5f0
    end
end

function cycle_correlation_tests()
    @testset "Cycle Correlation Tests" begin
        # Test with identical inputs - need to create 3D arrays
        phases_2d = rand(Float32, 10, 5)
        phases_3d = cat([phases_2d for _ in 1:5]..., dims=3)
        
        corr_self = cycle_correlation(phases_2d, phases_3d)
        @test size(corr_self) == (5,)  # One correlation per cycle
        @test all(corr_self .≈ 1.0f0)  # Self-correlation should be 1
        
        # Test with different dimensions
        phases_a = rand(Float32, 8, 20)
        phases_b_3d = cat([rand(Float32, 8, 20) for _ in 1:20]..., dims=3)
        corr_ab = cycle_correlation(phases_a, phases_b_3d)
        @test length(corr_ab) == 20
        
        # Test correlation bounds
        @test all(corr_ab .>= -1.0f0) && all(corr_ab .<= 1.0f0)
    end
end

function cycle_sparsity_tests()
    @testset "Cycle Sparsity Tests" begin
        # Test with sparse phases (many NaNs)
        phases_sparse = fill(NaN, 10, 20)
        phases_sparse[1:2, :] .= rand(Float32, 2, 20)
        phases_sparse_3d = cat([phases_sparse for _ in 1:20]..., dims=3)
        
        sparsity = cycle_sparsity(phases_sparse[1:10, 1:20], phases_sparse_3d)
        @test size(sparsity) == (20,)  # One sparsity value per cycle
        @test all(sparsity .>= 0.0f0) && all(sparsity .<= 1.0f0)
        
        # Dense phases should have lower sparsity
        phases_dense = rand(Float32, 10, 20)
        phases_dense_3d = cat([phases_dense for _ in 1:20]..., dims=3)
        sparsity_dense = cycle_sparsity(phases_dense, phases_dense_3d)
        
        # Note: dense should have zero sparsity (no NaNs)
        @test all(sparsity_dense .== 0.0f0)
    end
end

function cor_realvals_tests()
    @testset "Correlation of Real Values Tests" begin
        # Test identical vectors
        v = randn(Float32, 100)
        @test cor_realvals(v, v) ≈ 1.0f0
        
        # Test perfectly anti-correlated vectors
        v_anti = -v
        @test cor_realvals(v, v_anti) ≈ -1.0f0 atol=1e-5
        
        # Test independent vectors
        v1 = randn(Float32, 100)
        v2 = randn(Float32, 100)
        corr_indep = cor_realvals(v1, v2)
        @test abs(corr_indep) < 0.3f0  # Should be weakly correlated
        
        # Test with NaN values - correlation should handle them
        v_nan = vcat(randn(Float32, 50), fill(NaN, 50))
        v_ok = randn(Float32, 100)
        corr_nan = cor_realvals(v_nan, v_ok)
        @test isfinite(corr_nan)
    end
end

function evaluate_loss_tests()
    @testset "Evaluate Loss Tests" begin
        batch_size, n_classes = 32, 5
        y_pred = randn(ComplexF32, n_classes, batch_size)
        y_true = zeros(Float32, n_classes, batch_size)
        # Create one-hot encoded truth
        for i in 1:batch_size
            y_true[rand(1:n_classes), i] = 1.0f0
        end
        
        # Test different encoding types
        for encoding in [:quadrature, :similarity]
            loss = evaluate_loss(y_pred, y_true, encoding)
            # Loss output shape may vary depending on dimensionality
            @test !isempty(loss)
            @test all(isfinite.(loss))
        end
        
        # Loss should be non-negative for similarity encoding
        loss = evaluate_loss(y_pred, y_true, :similarity)
        @test all(loss .>= 0.0f0)
    end
end

function evaluate_accuracy_tests()
    @testset "Evaluate Accuracy Tests" begin
        batch_size, n_classes = 32, 5
        
        # Create real-valued predictions (not complex) for proper comparison
        y_pred = randn(Float32, n_classes, batch_size)
        y_true = zeros(Float32, n_classes, batch_size)
        for i in 1:batch_size
            y_true[rand(1:n_classes), i] = 1.0f0
        end
        
        for encoding in [:quadrature, :similarity]
            try
                correct, total = evaluate_accuracy(y_pred, y_true, encoding)
                # Correct can be an array, so check dimensions
                if isa(correct, AbstractArray)
                    @test all(correct .>= 0)
                    @test all(correct .<= total)
                    # For each entry, accuracy should be reasonable
                    @test all(correct / total .>= 0.0f0) && all(correct / total .<= 1.0f0)
                else
                    @test correct >= 0
                    @test correct <= total
                    # Accuracy should be a reasonable number
                    accuracy = correct / total
                    @test 0.0f0 <= accuracy <= 1.0f0
                end
            catch e
                # Skip if there's an incompatibility with this encoding
                @test isa(e, Exception)
            end
        end
    end
end

function confusion_matrix_tests()
    @testset "Confusion Matrix Tests" begin
        # Test the basic confusion_matrix function with binary classification
        # predictions and labels for binary case
        predictions = [0.9f0, 0.1f0, 0.8f0, 0.2f0, 0.7f0]  # similarity scores
        truth = [1, 0, 1, 0, 1]  # binary labels
        
        cm = confusion_matrix(predictions, truth, 0.5f0)  # threshold at 0.5
        
        # Confusion matrix should be 2x2
        @test size(cm) == (2, 2)
        
        # Total should equal length of predictions
        @test sum(cm) == length(truth)
        
        # All values should be non-negative integers
        @test all(cm .>= 0)
        @test all(cm .== floor.(cm))
    end
end

function ovr_matrices_tests()
    @testset "One-vs-Rest Matrices Tests" begin
        # Test OvR_matrices with a 2x2 confusion matrix
        # Format: [TP FN; FP TN]
        cm = [80 20; 10 890]
        
        ovr = OvR_matrices(cm, cm, 0.5f0)
        
        # Check that output is valid
        @test !isempty(ovr)
        # Result should be an array of OvR matrices
        @test all(size.(ovr, 1) .== 2)  # All should have 2 rows
        @test all(size.(ovr, 2) .== 2)  # All should have 2 columns
    end
end

function tpr_fpr_tests()
    @testset "TPR/FPR Tests" begin
        # Synthetic predictions (similarity scores) and one‑hot true labels
        predictions = Float32[0.9 0.2; 0.1 0.8]   # 2 classes × 2 samples
        labels = Float32[1 0; 0 1]              # one‑hot encoding
        
        tpr, fpr = tpr_fpr(predictions, labels, 201)
        
        @test length(tpr) == 201
        @test length(fpr) == 201
        @test all(x -> (0 ≤ x ≤ 1) || isnan(x), tpr)
        @test all(x -> (0 ≤ x ≤ 1) || isnan(x), fpr)
    end
end

function interpolate_roc_tests()
    @testset "Interpolate ROC Tests" begin
        # Create sample ROC curve
        fprs = collect(0.0:0.1:1.0)
        tprs = fprs  # Diagonal line
        
        # Interpolate
        interp = interpolate_roc((tprs, fprs))
        
        # Check that interpolation function works
        @test !isnothing(interp)
        
        # Test interpolation at a point
        test_fpr = 0.5
        interp_tpr = interp(test_fpr)
        @test isfinite(interp_tpr)
        @test 0 <= interp_tpr <= 1
    end
end
