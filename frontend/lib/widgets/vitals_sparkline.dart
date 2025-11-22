import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class VitalsSparkline extends StatelessWidget {
  const VitalsSparkline({
    super.key,
    required this.heartRate,
    required this.spo2,
  });

  final List<double> heartRate;
  final List<double> spo2;

  @override
  Widget build(BuildContext context) {
    if (heartRate.isEmpty && spo2.isEmpty) {
      return const SizedBox.shrink();
    }
    return SizedBox(
      height: 80,
      child: CustomPaint(
        painter: _VitalsSparklinePainter(
          heartRate: heartRate,
          spo2: spo2,
        ),
      ),
    );
  }
}

class _VitalsSparklinePainter extends CustomPainter {
  _VitalsSparklinePainter({
    required this.heartRate,
    required this.spo2,
  });

  final List<double> heartRate;
  final List<double> spo2;

  @override
  void paint(Canvas canvas, Size size) {
    final hr = heartRate;
    final s2 = spo2;

    final hrPaint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final spo2Paint = Paint()
      ..color = Colors.blueAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final axisPaint = Paint()
      ..color = Colors.grey.shade400
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    final baselineY = size.height * 0.5;
    canvas.drawLine(Offset(0, baselineY), Offset(size.width, baselineY), axisPaint);

    if (hr.isNotEmpty) {
      _drawSeries(canvas, size, hr, hrPaint);
    }
    if (s2.isNotEmpty) {
      _drawSeries(canvas, size, s2, spo2Paint);
    }
  }

  void _drawSeries(Canvas canvas, Size size, List<double> series, Paint paint) {
    if (series.length < 2) return;

    final minVal = series.reduce(math.min);
    final maxVal = series.reduce(math.max);
    final span = (maxVal - minVal).abs() < 1e-6 ? 1.0 : (maxVal - minVal);

    final dx = size.width / (series.length - 1);

    Offset pointFor(int i) {
      final x = dx * i;
      final norm = (series[i] - minVal) / span;
      final y = size.height * (1.0 - norm * 0.8 - 0.1);
      return Offset(x, y);
    }

    final path = Path()..moveTo(pointFor(0).dx, pointFor(0).dy);
    for (var i = 1; i < series.length; i++) {
      final p = pointFor(i);
      path.lineTo(p.dx, p.dy);
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _VitalsSparklinePainter oldDelegate) {
    return !listEquals(heartRate, oldDelegate.heartRate) ||
        !listEquals(spo2, oldDelegate.spo2);
  }
}
