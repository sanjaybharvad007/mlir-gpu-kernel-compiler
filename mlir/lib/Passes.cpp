//===- Passes.cpp - tj-fuse and tj-lower-to-linalg -------------------------===//
//
// tj-fuse: pattern-matches the linalg idiom the Python emitter produces for
//   gelu(x @ w + b)  (linalg.matmul -> linalg.generic add with a rank-1
//   broadcast -> linalg.generic unary activation) and replaces it with one
//   tj.fused_matmul, i.e. the same decision tensorjit's Python fusion pass
//   makes, expressed as a rewrite pattern on MLIR.
//
// tj-lower-to-linalg: expands tj.* back into linalg/tensor/arith/math so the
//   stock pipeline (bufferize -> loops -> llvm) can run.
//
#include "TensorJit/Passes.h"
#include "TensorJit/TensorJitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tj;

namespace {

//===----------------------------------------------------------------------===//
// helpers for recognising the Python emitter's linalg.generic shapes
//===----------------------------------------------------------------------===//

/// True if `g` is an elementwise generic with one block op of type OpT (plus yield).
template <typename OpT>
static bool isUnaryGeneric(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumResults() != 1) return false;
  if (!llvm::all_of(g.getIteratorTypesArray(), linalg::isParallelIterator)) return false;
  Block &b = g.getRegion().front();
  if (b.getOperations().size() != 2) return false;
  return isa<OpT>(b.front());
}

/// linalg.generic { addf(%x, %bias) } where %bias is rank-1 broadcast along the last dim.
static bool isBiasAdd(linalg::GenericOp g, Value &bias) {
  if (g.getNumDpsInputs() != 2 || g.getNumResults() != 1) return false;
  if (!llvm::all_of(g.getIteratorTypesArray(), linalg::isParallelIterator)) return false;
  Block &b = g.getRegion().front();
  if (b.getOperations().size() != 2 || !isa<arith::AddFOp>(b.front())) return false;
  Value second = g.getDpsInputOperand(1)->get();
  auto t = dyn_cast<RankedTensorType>(second.getType());
  if (!t || t.getRank() != 1) return false;
  // indexing map of the bias must select only the last iteration dim
  AffineMap m = g.getIndexingMapsArray()[1];
  if (m.getNumResults() != 1) return false;
  auto dim = dyn_cast<AffineDimExpr>(m.getResult(0));
  if (!dim || dim.getPosition() != m.getNumDims() - 1) return false;
  bias = second;
  return true;
}

/// Recognise the activation body: relu = maximumf(x, 0); silu = x * sigmoid(x) (two ops);
/// gelu = 0.5*x*(1+erf(x/sqrt2)) (several ops).  We key on the distinctive op.
static std::optional<Epilogue> activationOf(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumResults() != 1) return std::nullopt;
  if (!llvm::all_of(g.getIteratorTypesArray(), linalg::isParallelIterator)) return std::nullopt;
  Block &b = g.getRegion().front();
  bool hasErf = false, hasMax = false, hasExp = false, hasDiv = false;
  for (Operation &op : b) {
    hasErf |= isa<math::ErfOp>(op);
    hasMax |= isa<arith::MaximumFOp>(op);
    hasExp |= isa<math::ExpOp>(op);
    hasDiv |= isa<arith::DivFOp>(op);
  }
  if (hasErf) return Epilogue::Gelu;
  if (hasMax && b.getOperations().size() <= 3) return Epilogue::Relu;
  if (hasExp && hasDiv) return Epilogue::Silu;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// tj-fuse
//===----------------------------------------------------------------------===//

/// matmul (+ bias) (+ activation) -> tj.fused_matmul.  Matches from the
/// outermost consumer so a chain is folded in one rewrite.
struct FuseMatmulEpilogue : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp g, PatternRewriter &rewriter) const override {
    // Peel: activation? -> bias add? -> matmul
    Epilogue epi = Epilogue::None;
    linalg::GenericOp cur = g;
    Value bias;
    if (auto act = activationOf(cur)) {
      epi = *act;
      auto prod = cur.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
      if (prod && isBiasAdd(prod, bias)) {
        if (!prod->hasOneUse()) return failure();
        cur = prod;
      } else {
        cur = nullptr;
        if (auto mm = g.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>())
          return rewriteWith(g, mm, Value(), epi, rewriter);
        return failure();
      }
    } else if (!isBiasAdd(cur, bias)) {
      return failure();
    }
    auto mm = cur.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>();
    if (!mm || !mm->hasOneUse()) return failure();
    return rewriteWith(g, mm, bias, epi, rewriter);
  }

  LogicalResult rewriteWith(linalg::GenericOp root, linalg::MatmulOp mm, Value bias, Epilogue epi,
                            PatternRewriter &rewriter) const {
    // The matmul's init must be a zero fill (linalg.fill of 0 into tensor.empty).
    auto fill = mm.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
    if (!fill) return failure();
    auto cst = fill.getDpsInputOperand(0)->get().getDefiningOp<arith::ConstantOp>();
    if (!cst || !matchPattern(cst.getValue(), m_AnyZeroFloat())) return failure();
    Value a = mm.getDpsInputOperand(0)->get();
    Value b = mm.getDpsInputOperand(1)->get();
    bool transB = false;
    // Fold an explicit transpose of B into trans_b.
    if (auto tr = b.getDefiningOp<linalg::TransposeOp>()) {
      auto perm = tr.getPermutation();
      if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
        b = tr.getInput();
        transB = true;
      }
    }
    rewriter.replaceOpWithNewOp<FusedMatmulOp>(root, root.getResultTypes()[0], a, b, bias,
                                               EpilogueAttr::get(rewriter.getContext(), epi),
                                               rewriter.getBoolAttr(transB));
    return success();
  }
};

struct FusePass : PassWrapper<FusePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePass)
  StringRef getArgument() const final { return "tj-fuse"; }
  StringRef getDescription() const final { return "fuse linalg matmul + bias + activation into tj.fused_matmul"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<TensorJitDialect, linalg::LinalgDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulEpilogue>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// tj-lower-to-linalg
//===----------------------------------------------------------------------===//

static Value emitActivation(OpBuilder &b, Location loc, Value x, Epilogue epi) {
  Type t = x.getType();
  switch (epi) {
    case Epilogue::None:
      return x;
    case Epilogue::Relu: {
      Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 0.0));
      return b.create<arith::MaximumFOp>(loc, x, zero);
    }
    case Epilogue::Gelu: {
      Value half = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 0.5));
      Value k = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 0.70710678118654752));
      Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 1.0));
      Value erf = b.create<math::ErfOp>(loc, b.create<arith::MulFOp>(loc, x, k));
      Value p = b.create<arith::AddFOp>(loc, one, erf);
      return b.create<arith::MulFOp>(loc, b.create<arith::MulFOp>(loc, half, x), p);
    }
    case Epilogue::Silu: {
      Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 1.0));
      Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
      Value sig = b.create<arith::DivFOp>(loc, one, b.create<arith::AddFOp>(loc, one, e));
      return b.create<arith::MulFOp>(loc, x, sig);
    }
  }
  return x;
}

/// Elementwise linalg.generic over `shape` with inputs mapped by `maps`.
static Value elementwise(OpBuilder &b, Location loc, RankedTensorType resTy, ValueRange ins,
                         ArrayRef<AffineMap> inMaps,
                         function_ref<Value(OpBuilder &, Location, ValueRange)> body) {
  const int64_t rank = resTy.getRank();
  SmallVector<AffineMap> maps(inMaps.begin(), inMaps.end());
  maps.push_back(b.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  Value init = b.create<tensor::EmptyOp>(loc, resTy.getShape(), resTy.getElementType());
  auto g = b.create<linalg::GenericOp>(
      loc, resTy, ins, init, maps, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        nb.create<linalg::YieldOp>(nl, body(nb, nl, args.drop_back()));
      });
  return g.getResult(0);
}

/// Reduce over the last axis with `op` (sum or max), returning tensor<...xf>.
static Value reduceLast(OpBuilder &b, Location loc, Value x, bool isMax) {
  auto xt = cast<RankedTensorType>(x.getType());
  const int64_t rank = xt.getRank();
  auto resTy = RankedTensorType::get(xt.getShape().drop_back(), xt.getElementType());
  Value init = b.create<tensor::EmptyOp>(loc, resTy.getShape(), resTy.getElementType());
  Value seed = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(xt.getElementType(), isMax ? -3.40282347e38 : 0.0));
  Value filled = b.create<linalg::FillOp>(loc, seed, init).getResult(0);
  SmallVector<AffineMap> maps{b.getMultiDimIdentityMap(rank),
                              AffineMap::get(rank, 0, b.getMultiDimIdentityMap(rank).getResults().drop_back(), b.getContext())};
  SmallVector<utils::IteratorType> iters(rank - 1, utils::IteratorType::parallel);
  iters.push_back(utils::IteratorType::reduction);
  auto g = b.create<linalg::GenericOp>(
      loc, resTy, x, filled, maps, iters, [&](OpBuilder &nb, Location nl, ValueRange args) {
        Value r = isMax ? nb.create<arith::MaximumFOp>(nl, args[1], args[0]).getResult()
                        : nb.create<arith::AddFOp>(nl, args[1], args[0]).getResult();
        nb.create<linalg::YieldOp>(nl, r);
      });
  return g.getResult(0);
}

static AffineMap dropLastMap(OpBuilder &b, int64_t rank) {
  return AffineMap::get(rank, 0, b.getMultiDimIdentityMap(rank).getResults().drop_back(), b.getContext());
}
static AffineMap lastOnlyMap(OpBuilder &b, int64_t rank) {
  return AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext());
}

struct LowerSoftmax : OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SoftmaxOp op, PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    Value x = op.getInput();
    auto xt = cast<RankedTensorType>(x.getType());
    const int64_t rank = xt.getRank();
    Value m = reduceLast(rw, loc, x, /*isMax=*/true);
    Value e = elementwise(rw, loc, xt, {x, m}, {rw.getMultiDimIdentityMap(rank), dropLastMap(rw, rank)},
                          [](OpBuilder &b, Location l, ValueRange a) {
                            return b.create<math::ExpOp>(l, b.create<arith::SubFOp>(l, a[0], a[1])).getResult();
                          });
    Value s = reduceLast(rw, loc, e, /*isMax=*/false);
    Value y = elementwise(rw, loc, xt, {e, s}, {rw.getMultiDimIdentityMap(rank), dropLastMap(rw, rank)},
                          [](OpBuilder &b, Location l, ValueRange a) {
                            return b.create<arith::DivFOp>(l, a[0], a[1]).getResult();
                          });
    rw.replaceOp(op, y);
    return success();
  }
};

struct LowerLayerNorm : OpRewritePattern<LayerNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LayerNormOp op, PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    Value x = op.getInput();
    auto xt = cast<RankedTensorType>(x.getType());
    const int64_t rank = xt.getRank();
    const double invN = 1.0 / xt.getDimSize(rank - 1);
    Type et = xt.getElementType();
    auto scale = [&](Value v) {  // v / N
      auto vt = cast<RankedTensorType>(v.getType());
      return elementwise(rw, loc, vt, {v}, {rw.getMultiDimIdentityMap(vt.getRank())},
                         [&](OpBuilder &b, Location l, ValueRange a) {
                           Value c = b.create<arith::ConstantOp>(l, b.getFloatAttr(et, invN));
                           return b.create<arith::MulFOp>(l, a[0], c).getResult();
                         });
    };
    Value mean = scale(reduceLast(rw, loc, x, false));
    Value d = elementwise(rw, loc, xt, {x, mean}, {rw.getMultiDimIdentityMap(rank), dropLastMap(rw, rank)},
                          [](OpBuilder &b, Location l, ValueRange a) {
                            return b.create<arith::SubFOp>(l, a[0], a[1]).getResult();
                          });
    Value d2 = elementwise(rw, loc, xt, {d}, {rw.getMultiDimIdentityMap(rank)},
                           [](OpBuilder &b, Location l, ValueRange a) {
                             return b.create<arith::MulFOp>(l, a[0], a[0]).getResult();
                           });
    Value var = scale(reduceLast(rw, loc, d2, false));
    const float eps = op.getEps().convertToFloat();
    Value rstd = elementwise(rw, loc, cast<RankedTensorType>(var.getType()), {var},
                             {rw.getMultiDimIdentityMap(rank - 1)}, [&](OpBuilder &b, Location l, ValueRange a) {
                               Value e = b.create<arith::ConstantOp>(l, b.getFloatAttr(et, eps));
                               return b.create<math::RsqrtOp>(l, b.create<arith::AddFOp>(l, a[0], e)).getResult();
                             });
    SmallVector<Value> ins{d, rstd};
    SmallVector<AffineMap> maps{rw.getMultiDimIdentityMap(rank), dropLastMap(rw, rank)};
    if (op.getWeight()) { ins.push_back(op.getWeight()); maps.push_back(lastOnlyMap(rw, rank)); }
    if (op.getBias()) { ins.push_back(op.getBias()); maps.push_back(lastOnlyMap(rw, rank)); }
    const bool hasW = static_cast<bool>(op.getWeight()), hasB = static_cast<bool>(op.getBias());
    Value y = elementwise(rw, loc, xt, ins, maps, [&](OpBuilder &b, Location l, ValueRange a) {
      Value v = b.create<arith::MulFOp>(l, a[0], a[1]);
      unsigned i = 2;
      if (hasW) v = b.create<arith::MulFOp>(l, v, a[i++]);
      if (hasB) v = b.create<arith::AddFOp>(l, v, a[i++]);
      return v;
    });
    rw.replaceOp(op, y);
    return success();
  }
};

struct LowerRmsNorm : OpRewritePattern<RmsNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RmsNormOp op, PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    Value x = op.getInput();
    auto xt = cast<RankedTensorType>(x.getType());
    const int64_t rank = xt.getRank();
    Type et = xt.getElementType();
    const double invN = 1.0 / xt.getDimSize(rank - 1);
    const float eps = op.getEps().convertToFloat();
    Value x2 = elementwise(rw, loc, xt, {x}, {rw.getMultiDimIdentityMap(rank)},
                           [](OpBuilder &b, Location l, ValueRange a) {
                             return b.create<arith::MulFOp>(l, a[0], a[0]).getResult();
                           });
    Value ms = reduceLast(rw, loc, x2, false);
    Value rstd = elementwise(rw, loc, cast<RankedTensorType>(ms.getType()), {ms},
                             {rw.getMultiDimIdentityMap(rank - 1)}, [&](OpBuilder &b, Location l, ValueRange a) {
                               Value c = b.create<arith::ConstantOp>(l, b.getFloatAttr(et, invN));
                               Value e = b.create<arith::ConstantOp>(l, b.getFloatAttr(et, eps));
                               Value m = b.create<arith::AddFOp>(l, b.create<arith::MulFOp>(l, a[0], c), e);
                               return b.create<math::RsqrtOp>(l, m).getResult();
                             });
    SmallVector<Value> ins{x, rstd};
    SmallVector<AffineMap> maps{rw.getMultiDimIdentityMap(rank), dropLastMap(rw, rank)};
    const bool hasW = static_cast<bool>(op.getWeight());
    if (hasW) { ins.push_back(op.getWeight()); maps.push_back(lastOnlyMap(rw, rank)); }
    Value y = elementwise(rw, loc, xt, ins, maps, [&](OpBuilder &b, Location l, ValueRange a) {
      Value v = b.create<arith::MulFOp>(l, a[0], a[1]);
      if (hasW) v = b.create<arith::MulFOp>(l, v, a[2]);
      return v;
    });
    rw.replaceOp(op, y);
    return success();
  }
};

struct LowerFusedMatmul : OpRewritePattern<FusedMatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FusedMatmulOp op, PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    auto rt = cast<RankedTensorType>(op.getResult().getType());
    if (rt.getRank() != 2) return rw.notifyMatchFailure(op, "batched lowering not implemented");
    Type et = rt.getElementType();
    Value b = op.getRhs();
    if (op.getTransB()) {
      auto bt = cast<RankedTensorType>(b.getType());
      Value init = rw.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{bt.getDimSize(1), bt.getDimSize(0)}, et);
      b = rw.create<linalg::TransposeOp>(loc, b, init, ArrayRef<int64_t>{1, 0}).getResult()[0];
    }
    Value zero = rw.create<arith::ConstantOp>(loc, rw.getFloatAttr(et, 0.0));
    Value init = rw.create<tensor::EmptyOp>(loc, rt.getShape(), et);
    Value filled = rw.create<linalg::FillOp>(loc, zero, init).getResult(0);
    Value acc = rw.create<linalg::MatmulOp>(loc, rt, ValueRange{op.getLhs(), b}, filled).getResult(0);
    const Epilogue epi = op.getEpilogue();
    Value bias = op.getBias();
    if (!bias && epi == Epilogue::None) {
      rw.replaceOp(op, acc);
      return success();
    }
    SmallVector<Value> ins{acc};
    SmallVector<AffineMap> maps{rw.getMultiDimIdentityMap(2)};
    if (bias) { ins.push_back(bias); maps.push_back(lastOnlyMap(rw, 2)); }
    Value y = elementwise(rw, loc, rt, ins, maps, [&](OpBuilder &nb, Location l, ValueRange a) {
      Value v = a[0];
      if (bias) v = nb.create<arith::AddFOp>(l, v, a[1]);
      return emitActivation(nb, l, v, epi);
    });
    rw.replaceOp(op, y);
    return success();
  }
};

struct LowerToLinalgPass : PassWrapper<LowerToLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)
  StringRef getArgument() const final { return "tj-lower-to-linalg"; }
  StringRef getDescription() const final { return "lower tj.* ops to linalg/tensor/arith/math"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect, math::MathDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSoftmax, LowerLayerNorm, LowerRmsNorm, LowerFusedMatmul>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::tj::createFusePass() { return std::make_unique<FusePass>(); }
std::unique_ptr<Pass> mlir::tj::createLowerToLinalgPass() { return std::make_unique<LowerToLinalgPass>(); }

void mlir::tj::registerPasses() {
  PassRegistration<FusePass>();
  PassRegistration<LowerToLinalgPass>();
}
