import 'dart:convert';

import 'package:http/http.dart' as http;

/// Default API base URL.
///
/// This can be overridden at build/run time with:
///   --dart-define=API_BASE_URL=http://<your-mac-ip>:8000
const String kApiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://192.168.0.141:8000',
);

class ApiClient {
  ApiClient({
    this.baseUrl = kApiBaseUrl,
  });

  final String baseUrl;

  Future<List<Map<String, dynamic>>> listCases() async {
    final uri = Uri.parse('$baseUrl/cases/');
    final res = await http.get(uri).timeout(
      const Duration(seconds: 10),
      onTimeout: () {
        throw Exception('Failed to load cases: Request timed out');
      },
    );
    if (res.statusCode != 200) {
      throw Exception('Failed to list cases: ${res.body}');
    }
    final decoded = jsonDecode(res.body) as List<dynamic>;
    return decoded.cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> getCaseDetail(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to get case detail: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<String> createCase({
    required String patientId,
    String? notes,
  }) async {
    final uri = Uri.parse('$baseUrl/cases/');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'patient_id': patientId,
        'notes': notes,
      }),
    ).timeout(
      const Duration(seconds: 30),
      onTimeout: () {
        throw Exception('Request timed out after 30 seconds. Please check your connection.');
      },
    );
    if (res.statusCode != 200) {
      throw Exception('Failed to create case: ${res.body}');
    }
    return res.body.replaceAll('"', '');
  }

  Future<void> addSymptoms({
    required String caseId,
    required String text,
    int topN = 5,
  }) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/symptoms');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'top_n': topN,
      }),
    ).timeout(
      const Duration(seconds: 60),
      onTimeout: () {
        throw Exception('Symptom analysis timed out after 60 seconds. The NLP model may be loading for the first time.');
      },
    );
    if (res.statusCode != 200) {
      throw Exception('Failed to add symptoms: ${res.body}');
    }
  }

  Future<void> addVitals({
    required String caseId,
    required List<double> heartRate,
    required List<double> spo2,
    required List<double> temperature,
    required List<double> respRate,
  }) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/vitals');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'heart_rate': heartRate,
        'spo2': spo2,
        'temperature': temperature,
        'resp_rate': respRate,
      }),
    );
    if (res.statusCode != 200) {
      throw Exception('Failed to add vitals: ${res.body}');
    }
  }

  Future<Map<String, dynamic>> getAnalysis(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/analysis');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to get analysis: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> getTimeline(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/timeline');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to get timeline: ${res.body}');
    }
    final decoded = jsonDecode(res.body) as List<dynamic>;
    return decoded.cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> getAgentTimeline(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/agent-timeline');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to get agent timeline: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> getReport(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/report');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to get report: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  Future<void> uploadLabReport({
    required String caseId,
    required String filePath,
  }) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/upload-report');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    final streamed = await request.send();
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 200) {
      throw Exception('Failed to upload lab report: ${res.body}');
    }
  }

  Future<void> uploadImage({
    required String caseId,
    required String filePath,
  }) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId/upload-image');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    final streamed = await request.send();
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 200) {
      throw Exception('Failed to upload image: ${res.body}');
    }
  }

  Future<Map<String, dynamic>> trainLocal({
    String modelName = 'multimodal_fusion',
    int epochs = 1,
    double learningRate = 1e-2,
  }) async {
    final uri = Uri.parse('$baseUrl/train/local');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'model_name': modelName,
        'epochs': epochs,
        'learning_rate': learningRate,
      }),
    );
    if (res.statusCode != 200) {
      throw Exception('Failed to run local training: ${res.body}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }
  Future<void> deleteCase(String caseId) async {
    final uri = Uri.parse('$baseUrl/cases/$caseId');
    final res = await http.delete(uri);

    if (res.statusCode != 204) {
      throw Exception('Failed to delete case: ${res.body}');
    }
  }

  Future<List<Map<String, dynamic>>> searchMedicines(String query) async {
    final uri = Uri.parse('$baseUrl/medicines/search?q=$query');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Failed to search medicines: ${res.body}');
    }
    final decoded = jsonDecode(res.body) as List<dynamic>;
    return decoded.cast<Map<String, dynamic>>();
  }
}
