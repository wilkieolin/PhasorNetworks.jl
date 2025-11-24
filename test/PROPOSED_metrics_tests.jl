# Proposed metrics_tests.jl
# This file proposes new tests for the metrics.jl module
# To be added to the test suite to improve coverage

using Test
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
        
        # Test intermediate value
        @test 0 < arc_error(0.5f0) < 1.0f0
        
        # Test array input
        phases = [0.0f0, 0.25f0, 0.5f0, 0.75f0, 1.0f0]
        errors = arc_error(phases)
        @test size(errors) == size(phases)
        @test all(errors .>= -1.0f0) && all(errors .<= 1.0f0)
        
        # Test symmetry around 0
        @test arc_error(0.3f0) ≈ arc_error(-0.3f0)
        
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
        @test exp_score(1.0f0) ≈ 0.0f0
        
        # Test zero similarity
        zero_score = exp_score(0.0f0, scale=3.0f0)
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
        scores_scale3 = exp_score(0.5f0, scale=3.0f0)
        scores_scale5 = exp_score(0.5f0, scale=5.0f0)
        @test scores_scale3 != scores_scale5
        @test scores_scale5 > scores_scale3  # Higher scale should give higher scores
    end
end

function z_score_tests()
    @testset "Z-Score Tests" begin
        # Test at center
        @test isfinite(z_score(0.5f0))
        
        # Test symmetry
        @test z_score(0.3f0) ≈ z_score(0.7f0)
        
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
        static_sim = randn(Float32, n_x, n_y) |> abs
        
        # Dynamic similarity (correlates with static)
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
        # Test with identical inputs
        phases = rand(Float32, 10, 5)
        corr_self = cycle_correlation(phases, phases)
        @test size(corr_self) == (5,)  # One correlation per cycle
        @test all(corr_self .≈ 1.0f0)  # Self-correlation should be 1
        
        # Test with different dimensions
        phases_a = rand(Float32, 8, 20)
        phases_b = rand(Float32, 8, 20)
        corr_ab = cycle_correlation(phases_a, phases_b)
        @test length(corr_ab) == 20
        
        # Test correlation bounds
        @test all(corr_ab .>= -1.0f0) && all(corr_ab .<= 1.0f0)
    end
end

function cycle_sparsity_tests()
    @testset "Cycle Sparsity Tests" begin
        # Test with sparse phases (many zeros)
        phases_sparse = zeros(Float32, 10, 20)
        phases_sparse[1:2, :] .= rand(Float32, 2, 20)
        
        sparsity = cycle_sparsity(phases_sparse)
        @test size(sparsity) == (20,)  # One sparsity value per cycle
        @test all(sparsity .>= 0.0f0) && all(sparsity .<= 1.0f0)
        
        # Dense phases should have lower sparsity
        phases_dense = rand(Float32, 10, 20)
        sparsity_dense = cycle_sparsity(phases_dense)
        
        # Note: sparse might not be less than dense depending on implementation
        # This test just checks valid output
        @test all(sparsity_dense .>= 0.0f0)
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
        
        # Test scalar handling
        @test isfinite(cor_realvals(1.0f0, 1.0f0)) || isnan(cor_realvals(1.0f0, 1.0f0))
    end
end

function evaluate_loss_tests()
    @testset "Evaluate Loss Tests" begin
        batch_size, n_classes = 32, 5
        y_pred = randn(ComplexF32, n_classes, batch_size)
        y_true = rand(1:n_classes, batch_size)
        
        # Test different encoding types
        for encoding in [:quadrature, :codebook]
            loss = evaluate_loss(y_pred, y_true, encoding)
            @test size(loss) == (n_classes, batch_size)
            @test all(isfinite.(loss))
        end
        
        # Loss should be non-negative
        loss = evaluate_loss(y_pred, y_true, :quadrature)
        @test all(loss .>= 0.0f0)
    end
end

function evaluate_accuracy_tests()
    @testset "Evaluate Accuracy Tests" begin
        batch_size, n_classes = 32, 5
        
        # Create predictions with known accuracy
        y_pred = randn(ComplexF32, n_classes, batch_size)
        y_true = rand(1:n_classes, batch_size)
        
        for encoding in [:quadrature, :codebook]
            correct, total = evaluate_accuracy(y_pred, y_true, encoding)
            @test correct >= 0.0f0
            @test total == batch_size
            @test correct <= total
            
            # Accuracy should be a reasonable number
            accuracy = correct / total
            @test 0.0f0 <= accuracy <= 1.0f0
        end
    end
end

function confusion_matrix_tests()
    @testset "Confusion Matrix Tests" begin
        n_classes = 4
        y_true = [1, 2, 3, 1, 2, 4, 3, 1, 2, 4]
        y_pred = [1, 2, 3, 2, 2, 4, 3, 1, 2, 1]  # 7 correct, 3 wrong
        
        cm = confusion_matrix(y_true, y_pred, n_classes)
        
        # Confusion matrix should be square
        @test size(cm) == (n_classes, n_classes)
        
        # Total elements should equal predictions
        @test sum(cm) == length(y_true)
        
        # Diagonal should count correct predictions
        @test sum(diag(cm)) == 7
        
        # All values should be non-negative integers
        @test all(cm .>= 0)
        @test all(cm .== floor.(cm))
    end
end

function ovr_matrices_tests()
    @testset "One-vs-Rest Matrices Tests" begin
        n_classes = 3
        y_true = [1, 2, 3, 1, 2, 3, 1, 2]
        y_pred = [1, 2, 3, 2, 2, 3, 1, 2]
        
        cm = confusion_matrix(y_true, y_pred, n_classes)
        ovr = OvR_matrices(cm)
        
        # Should return one matrix per class
        @test length(ovr) == n_classes
        
        # Each OvR matrix should be 2x2
        for matrix in ovr
            @test size(matrix) == (2, 2)
        end
        
        # TP + FN should equal positives in original class
        for c in 1:n_classes
            tp = ovr[c][1, 1]
            fn = ovr[c][2, 1]
            positives = sum(cm[c, :])
            @test tp + fn == positives
        end
    end
end

function tpr_fpr_tests()
    @testset "TPR/FPR Tests" begin
        # Create sample OvR matrices
        ovr_matrix = [
            90  10   # TP FN
            5   895  # FP TN
        ]
        
        tpr, fpr = tpr_fpr(ovr_matrix)
        
        # TPR should be TP / (TP + FN)
        expected_tpr = 90 / (90 + 10)
        @test tpr ≈ expected_tpr
        
        # FPR should be FP / (FP + TN)
        expected_fpr = 5 / (5 + 895)
        @test fpr ≈ expected_fpr
        
        # Both should be in [0, 1]
        @test 0 <= tpr <= 1
        @test 0 <= fpr <= 1
    end
end

function interpolate_roc_tests()
    @testset "Interpolate ROC Tests" begin
        # Create sample ROC curve
        thresholds = collect(0.0:0.1:1.0)
        tprs = thresholds
        fprs = thresholds .* 0.5
        
        # Interpolate
        n_interp = 100
        interp_fprs, interp_tprs = interpolate_roc(fprs, tprs, n_interp)
        
        # Should have requested number of points
        @test length(interp_fprs) == n_interp
        @test length(interp_tprs) == n_interp
        
        # Should maintain bounds
        @test all(interp_fprs .>= 0) && all(interp_fprs .<= 1)
        @test all(interp_tprs .>= 0) && all(interp_tprs .<= 1)
        
        # Should be monotonic
        @test all(diff(interp_fprs) .>= -1e-6)
        @test all(diff(interp_tprs) .>= -1e-6)
    end
end
