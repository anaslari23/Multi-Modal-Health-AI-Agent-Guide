import 'dart:math' as math;

import 'package:flutter/material.dart';

class ModalityRadarChart extends StatelessWidget {
  const ModalityRadarChart({
    super.key,
    required this.nlpScore,
    required this.imagingScore,
    required this.labScore,
    required this.vitalsScore,
  });

  final double nlpScore;
  final double imagingScore;
  final double labScore;
  final double vitalsScore;

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: CustomPaint(
        painter: _ModalityRadarPainter(
          nlpScore: nlpScore.clamp(0.0, 1.0),
          imagingScore: imagingScore.clamp(0.0, 1.0),
          labScore: labScore.clamp(0.0, 1.0),
          vitalsScore: vitalsScore.clamp(0.0, 1.0),
        ),
      ),
    );
  }
}

class _ModalityRadarPainter extends CustomPainter {
  _ModalityRadarPainter({
    required this.nlpScore,
    required this.imagingScore,
    required this.labScore,
    required this.vitalsScore,
  });

  final double nlpScore;
  final double imagingScore;
  final double labScore;
  final double vitalsScore;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = math.min(size.width, size.height) * 0.35;

    final axisPaint = Paint()
      ..color = Colors.grey.shade400
      ..strokeWidth = 1;

    final outlinePaint = Paint()
      ..color = Colors.grey.shade300
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = Colors.blue.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    final borderPaint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    // Draw concentric circles for scale.
    for (var i = 1; i <= 3; i++) {
      final r = radius * (i / 3);
      canvas.drawCircle(center, r, outlinePaint);
    }

    // Angles for 4 modalities (starting at top, clockwise).
    final angles = <double>[
      -math.pi / 2, // NLP
      0, // Imaging
      math.pi / 2, // Labs
      math.pi, // Vitals
    ];

    // Draw axes.
    for (final angle in angles) {
      final dx = center.dx + radius * math.cos(angle);
      final dy = center.dy + radius * math.sin(angle);
      canvas.drawLine(center, Offset(dx, dy), axisPaint);
    }

    // Compute vertices for the scores.
    final scores = [nlpScore, imagingScore, labScore, vitalsScore];
    final points = <Offset>[];
    for (var i = 0; i < angles.length; i++) {
      final r = radius * scores[i];
      final dx = center.dx + r * math.cos(angles[i]);
      final dy = center.dy + r * math.sin(angles[i]);
      points.add(Offset(dx, dy));
    }

    // Draw filled polygon.
    final path = Path()..moveTo(points[0].dx, points[0].dy);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx, points[i].dy);
    }
    path.close();

    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, borderPaint);

    // Draw labels around the chart.
    final textPainter = (String text) {
      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(fontSize: 10, color: Colors.black87),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      return tp;
    };

    final labelOffset = radius + 16;
    final labels = ['NLP', 'Imaging', 'Labs', 'Vitals'];
    for (var i = 0; i < angles.length; i++) {
      final dx = center.dx + labelOffset * math.cos(angles[i]);
      final dy = center.dy + labelOffset * math.sin(angles[i]);
      final tp = textPainter(labels[i]);
      canvas.save();
      canvas.translate(dx - tp.width / 2, dy - tp.height / 2);
      tp.paint(canvas, Offset.zero);
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(covariant _ModalityRadarPainter oldDelegate) {
    return nlpScore != oldDelegate.nlpScore ||
        imagingScore != oldDelegate.imagingScore ||
        labScore != oldDelegate.labScore ||
        vitalsScore != oldDelegate.vitalsScore;
  }
}
