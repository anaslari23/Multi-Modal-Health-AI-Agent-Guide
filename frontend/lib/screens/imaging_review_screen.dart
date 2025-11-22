import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/api_client.dart';

class ImagingReviewScreen extends StatefulWidget {
  const ImagingReviewScreen({super.key});

  static const routeName = '/imaging';

  @override
  State<ImagingReviewScreen> createState() => _ImagingReviewScreenState();
}

class _ImagingReviewScreenState extends State<ImagingReviewScreen> {
  // Viewer controls
  double _brightness = 1.0;
  bool _showOverlay = true;
  double _overlayOpacity = 0.75;

  String? _caseId;
  Map<String, dynamic>? _imaging;
  Map<String, dynamic>? _xai;
  bool _loading = false;
  String? _error;

  // Simple brightness matrix for ColorFilter.matrix.
  // Scales RGB channels by [brightness] while keeping alpha unchanged.
  List<double> _brightnessMatrix(double b) {
    return <double>[
      b, 0, 0, 0, 0,
      0, b, 0, 0, 0,
      0, 0, b, 0, 0,
      0, 0, 0, 1, 0,
    ];
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _caseId ??= ModalRoute.of(context)?.settings.arguments as String?;
    if (_caseId != null && _imaging == null && !_loading) {
      _fetchAnalysis();
    }
  }

  Future<void> _uploadImaging() async {
    if (_caseId == null) return;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: ImageSource.gallery);
      if (picked == null) {
        return;
      }
      final api = ApiClient();
      await api.uploadImage(caseId: _caseId!, filePath: picked.path);
      await _fetchAnalysis();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to upload imaging: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  Future<void> _fetchAnalysis() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = ApiClient();
      final data = await api.getAnalysis(_caseId!);
      if (!mounted) return;
      setState(() {
        _imaging = data['imaging'] as Map<String, dynamic>?;
        _xai = data['xai'] as Map<String, dynamic>?;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load imaging analysis: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final gradcamPath = _imaging?['gradcam_path'] as String?;
    final gradcamUrl = gradcamPath == null
        ? null
        : 'http://localhost:8000/static/$gradcamPath';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Clinical Image Review'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: Container(
                  color: Colors.black,
                  child: gradcamUrl == null
                      ? const Center(
                          child: Icon(
                            Icons.image,
                            size: 120,
                            color: Colors.white24,
                          ),
                        )
                      : InteractiveViewer(
                          minScale: 0.8,
                          maxScale: 4.0,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              // Base image with brightness adjustment.
                              ColorFiltered(
                                colorFilter: ColorFilter.matrix(_brightnessMatrix(_brightness)),
                                child: Image.network(
                                  gradcamUrl,
                                  fit: BoxFit.contain,
                                  errorBuilder: (_, __, ___) => const Center(
                                    child: Text(
                                      'Failed to load image',
                                      style: TextStyle(color: Colors.white70),
                                    ),
                                  ),
                                ),
                              ),
                              // Optional overlay (for now we reuse Grad-CAM image with adjustable opacity).
                              if (_showOverlay)
                                Opacity(
                                  opacity: _overlayOpacity,
                                  child: Image.network(
                                    gradcamUrl,
                                    fit: BoxFit.contain,
                                  ),
                                ),
                            ],
                          ),
                        ),
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Column(
              children: [
                if (_caseId != null)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Case: $_caseId',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                if (_loading) const LinearProgressIndicator(),
                if (_error != null) ...[
                  const SizedBox(height: 8),
                  Text(_error!, style: const TextStyle(color: Colors.red)),
                ],
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Text('Brightness'),
                    Expanded(
                      child: Slider(
                        value: _brightness,
                        min: 0.5,
                        max: 1.5,
                        onChanged: (v) => setState(() => _brightness = v),
                      ),
                    ),
                  ],
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Grad-CAM overlay'),
                    Switch(
                      value: _showOverlay,
                      onChanged: gradcamUrl == null
                          ? null
                          : (v) => setState(() => _showOverlay = v),
                    ),
                  ],
                ),
                Row(
                  children: [
                    const Text('Overlay opacity'),
                    Expanded(
                      child: Slider(
                        value: _overlayOpacity,
                        min: 0.0,
                        max: 1.0,
                        onChanged: (v) => setState(() => _overlayOpacity = v),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                _AIInsightsCard(
                  imaging: _imaging,
                  xai: _xai,
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: _caseId == null ? null : _uploadImaging,
                    icon: const Icon(Icons.upload_file_outlined),
                    label: const Text('Upload Imaging & Refresh'),
                  ),
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _caseId == null
                        ? null
                        : () {
                            Navigator.pushNamed(
                              context,
                              '/report',
                              arguments: _caseId,
                            );
                          },
                    child: const Text('Save & Export Report'),
                  ),
                ),
                const SizedBox(height: 12),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _AIInsightsCard extends StatelessWidget {
  const _AIInsightsCard({
    this.imaging,
    this.xai,
  });

  final Map<String, dynamic>? imaging;
  final Map<String, dynamic>? xai;

  @override
  Widget build(BuildContext context) {
    final probs = (imaging?['probabilities'] as Map<String, dynamic>?) ?? {};
    final gradcamPath = imaging?['gradcam_path'] as String?;
    final summary = xai?['summary'] as String?;

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'AI Insights',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            if (probs.isEmpty)
              const Text('Imaging probabilities not available yet.'),
            for (final entry in probs.entries)
              Text('• ${entry.key}: ${(entry.value as num).toDouble().toStringAsFixed(2)}'),
            if (gradcamPath != null) ...[
              const SizedBox(height: 8),
              Text('Grad-CAM: $gradcamPath'),
            ],
            if (summary != null) ...[
              const SizedBox(height: 8),
              Text('Explanation: $summary'),
            ],
          ],
        ),
      ),
    );
  }
}
